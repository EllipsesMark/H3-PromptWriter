"""ComfyUI nodes: LLM/VLM-backed prompt writers for MiniMax H3.

A local stand-in for the closed H3-Context-IR module. The writer nodes turn
casual intent + the actual reference media into the structured prompts that
the native MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo nodes expect.
"""

import os

from . import client, guides, imaging, local_llm, validate

try:
    import folder_paths
    _MODELS_DIR = folder_paths.models_dir
except ImportError:  # selftest / outside ComfyUI
    _MODELS_DIR = None

FPS = 24


def _gguf_files():
    """Scan models/llm_gguf and models/LLM for GGUF files.

    Returns (model_names, mmproj_names) as paths relative to models/.
    """
    models, mmprojs = [], []
    if _MODELS_DIR:
        for sub in ("llm_gguf", "LLM"):
            root = os.path.join(_MODELS_DIR, sub)
            if not os.path.isdir(root):
                continue
            for dirpath, _, files in os.walk(root):
                for f in sorted(files):
                    if not f.lower().endswith(".gguf"):
                        continue
                    rel = os.path.relpath(os.path.join(dirpath, f),
                                          _MODELS_DIR)
                    (mmprojs if "mmproj" in f.lower() else models).append(rel)
    return models or ["(no gguf found in models/llm_gguf)"], \
        ["none"] + mmprojs


def align_frame_count(n):
    # mirror of comfy_extras/nodes_minimax_h3.py: snap up to the 17k+5 grid
    while n % 17 != 5:
        n += 1
    return n


def duration_seconds(length):
    return align_frame_count(max(5, length)) / FPS


BACKEND_INPUTS = {
    "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0,
                              "step": 0.05}),
    "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                     "tooltip": "Changing the seed re-runs the writer "
                     "(and is passed to backends that support it)."}),
    "retry_on_invalid": ("BOOLEAN", {"default": True,
                                     "tooltip": "If format validation fails, "
                                     "send the issues back to the LLM for one "
                                     "correction pass."}),
}

# HTTP fallback, used only when no H3_LLM loader is connected.
HTTP_INPUTS = {
    "base_url": ("STRING", {"default": "http://127.0.0.1:1234/v1",
                            "tooltip": "Only used when no llm loader is "
                            "connected. OpenAI-compatible endpoint (LM Studio "
                            ":1234/v1, llama.cpp server :8080/v1, Ollama "
                            ":11434/v1, or a cloud URL)."}),
    "model": ("STRING", {"default": "local",
                         "tooltip": "Model name for the HTTP backend."}),
    "api_key": ("STRING", {"default": "none",
                           "tooltip": "'none' for local servers."}),
}


def _make_chat_fn(llm, base_url, api_key, model, temperature, seed):
    """Internal llama.cpp when a loader is connected, HTTP otherwise."""
    if llm is not None:
        return lambda msgs: local_llm.chat_local(
            llm, msgs, temperature=temperature, seed=seed)
    return lambda msgs: client.chat(base_url, api_key, model, msgs,
                                    temperature=temperature, seed=seed)


def _system_for(default_text, override_key, instructions):
    """Resolve the system prompt: built-in guide by default, optionally
    replaced wholesale and/or extended with user session directives."""
    if not instructions:
        return default_text
    text = (instructions.get(override_key) or "").strip() or default_text
    extra = (instructions.get("extra") or "").strip()
    if extra:
        text += (
            "\n\nAdditional session directives from the user. These are "
            "authoritative: where they conflict with anything above, the "
            "directives below win. The output contract (only the final "
            "prompt text, no commentary) still applies.\n" + extra)
    return text


def _run_with_retry(messages, issues_fn, chat_fn, retry):
    text = validate.strip_fences(chat_fn(messages))
    issues = issues_fn(text)
    attempts = 1
    if issues and retry:
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content":
                "Your output has format problems:\n- "
                + "\n- ".join(issues)
                + "\nRewrite the COMPLETE prompt with these fixed. "
                  "Output only the corrected prompt text."},
        ]
        text2 = validate.strip_fences(chat_fn(messages))
        issues2 = issues_fn(text2)
        attempts = 2
        if len(issues2) < len(issues):
            text, issues = text2, issues2
    return text, issues, attempts


class H3PromptLLMLoader:
    """Configure an in-process GGUF (+ mmproj for vision) for the writers.

    Loading is lazy: nothing is loaded until a writer runs. With keep_loaded
    off, the model is freed right after each prompt so VRAM returns to H3.
    """

    CATEGORY = "MiniMax-H3/prompting"
    FUNCTION = "configure"
    RETURN_TYPES = ("H3_LLM",)
    RETURN_NAMES = ("llm",)

    @classmethod
    def INPUT_TYPES(cls):
        models, mmprojs = _gguf_files()
        return {
            "required": {
                "model_name": (models,),
                "mmproj_name": (mmprojs, {"tooltip":
                    "Vision projector matching the model. Required for the "
                    "writers to see reference images; 'none' = text-only."}),
                "n_ctx": ("INT", {"default": 16384, "min": 4096,
                                  "max": 131072, "step": 1024,
                                  "tooltip": "Context window. The system "
                                  "guide is ~4k tokens and each image adds "
                                  "more; 16384 is a safe floor."}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999,
                                         "tooltip": "-1 = fully on GPU."}),
                "keep_loaded": ("BOOLEAN", {"default": False, "tooltip":
                    "Keep the LLM in VRAM between runs. Off = free it after "
                    "each prompt so H3 sampling gets the memory back."}),
            },
        }

    def configure(self, model_name, mmproj_name, n_ctx, n_gpu_layers,
                  keep_loaded):
        if _MODELS_DIR is None:
            raise RuntimeError("ComfyUI folder_paths unavailable.")
        model_path = os.path.join(_MODELS_DIR, model_name)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"GGUF not found: {model_path}")
        mmproj_path = None
        if mmproj_name != "none":
            mmproj_path = os.path.join(_MODELS_DIR, mmproj_name)
            if not os.path.isfile(mmproj_path):
                raise FileNotFoundError(f"mmproj not found: {mmproj_path}")
        return ({"model_path": model_path, "mmproj_path": mmproj_path,
                 "n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers,
                 "keep_loaded": keep_loaded},)


class H3PromptInstructions:
    """Customize how the writer LLM is instructed.

    Leave everything empty for stock behavior. extra_instructions is appended
    to the built-in guide as high-priority session directives; the override
    fields replace the built-in guide entirely for the matching writer. The
    default guide texts are exposed as outputs so you can view them (Show Any
    / Show Text), copy one, tweak it, and paste it into an override field.
    """

    CATEGORY = "MiniMax-H3/prompting"
    FUNCTION = "configure"
    RETURN_TYPES = ("H3_INSTRUCTIONS", "STRING", "STRING")
    RETURN_NAMES = ("instructions", "default_keyframes_guide",
                    "default_reference_guide")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "extra_instructions": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Appended to the built-in guide as session "
                    "directives that take priority over the defaults: tone, "
                    "style conventions, content-handling rules, vocabulary, "
                    "pacing preferences, etc. Applies to both writers."}),
            },
            "optional": {
                "override_keyframes_guide": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Advanced: REPLACES the entire built-in "
                    "T2V/I2V/FL2V system guide. The validator still expects "
                    "the H3 format (alignment line + the three core fields), "
                    "so keep those requirements in your custom guide. Empty "
                    "= use the built-in guide."}),
                "override_reference_guide": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Advanced: REPLACES the entire built-in "
                    "Ref2VA system guide. The validator still expects the "
                    "six sections and fixed asset labels, so keep those "
                    "requirements in your custom guide. Empty = use the "
                    "built-in guide."}),
            },
        }

    def configure(self, extra_instructions,
                  override_keyframes_guide="", override_reference_guide=""):
        return ({"extra": extra_instructions,
                 "override_base": override_keyframes_guide,
                 "override_ref": override_reference_guide},
                guides.BASE_SYSTEM, guides.REF_SYSTEM)


def _report(mode, dur, issues, attempts):
    lines = [f"mode: {mode}", f"duration: {dur:.2f}s",
             f"llm attempts: {attempts}"]
    if issues:
        lines.append("VALIDATION ISSUES:")
        lines += [f"- {i}" for i in issues]
    else:
        lines.append("validation: OK")
    return "\n".join(lines)


class H3PromptWriterKeyframes:
    """T2VA / I2VA / FL2VA / L2VA writer for the FL2VA checkpoint.

    Mode is derived from which keyframes are connected:
    none -> t2va, first -> i2va, both -> fl2va, last only -> l2va.
    """

    CATEGORY = "MiniMax-H3/prompting"
    FUNCTION = "write"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "intent": ("STRING", {"multiline": True, "default":
                    "The subject slowly turns toward the camera and smiles.",
                    "tooltip": "What should happen: action, camera, mood, "
                    "any dialogue lines (verbatim), music wishes."}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600,
                                   "step": 17,
                                   "tooltip": "Frame count at 24 fps; use the "
                                   "same value as the MiniMax H3 node. "
                                   "124 = ~5s."}),
                **BACKEND_INPUTS,
            },
            "optional": {
                "llm": ("H3_LLM", {"tooltip":
                    "Connect an H3 Prompt LLM Loader to run the writer "
                    "inside ComfyUI. Leave empty to use the HTTP fields."}),
                "instructions": ("H3_INSTRUCTIONS", {"tooltip":
                    "Optional H3 Prompt Instructions node to customize how "
                    "the LLM is guided. Empty = built-in defaults."}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                **HTTP_INPUTS,
            },
        }

    def write(self, intent, length, temperature, seed, retry_on_invalid,
              llm=None, instructions=None, first_frame=None, last_frame=None,
              base_url="http://127.0.0.1:1234/v1", model="local",
              api_key="none"):
        dur = duration_seconds(length)
        if first_frame is not None and last_frame is not None:
            mode = "fl2va"
        elif first_frame is not None:
            mode = "i2va"
        elif last_frame is not None:
            mode = "l2va"
        else:
            mode = "t2va"

        parts = [client.text_part(
            f"Task mode: {mode.upper()}\n"
            f"Video duration: {dur:.2f} seconds "
            f"({align_frame_count(max(5, length))} frames at 24 fps).\n"
            f"User intent:\n{intent.strip()}")]
        if first_frame is not None:
            parts.append(client.text_part(
                "This image is <Picture 1>, the FIRST frame of the video:"))
            parts.append(client.image_part(
                imaging.tensor_frame_to_b64(first_frame[0])))
        if last_frame is not None:
            n = "2" if first_frame is not None else "1"
            parts.append(client.text_part(
                f"This image is <Picture {n}>, the LAST frame of the video:"))
            parts.append(client.image_part(
                imaging.tensor_frame_to_b64(last_frame[0])))

        system = _system_for(guides.BASE_SYSTEM, "override_base",
                             instructions)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": parts}]
        chat_fn = _make_chat_fn(llm, base_url, api_key, model,
                                temperature, seed)
        text, issues, attempts = _run_with_retry(
            messages, lambda t: validate.validate_base(t, mode, dur),
            chat_fn, retry_on_invalid)
        return (text, _report(mode, dur, issues, attempts))


class H3PromptWriterReference:
    """Six-section full-reference writer for the Ref2VA checkpoint.

    Connect the same media here as on MiniMaxH3ReferenceToVideo, in the same
    slots, so the <Picture i>/<Video k>/<Audio j> labels line up. Audio labels
    are computed from the soundtrack flags: each video's soundtrack label
    precedes the video, standalone audio comes last.
    """

    CATEGORY = "MiniMax-H3/prompting"
    FUNCTION = "write"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "report")

    @classmethod
    def INPUT_TYPES(cls):
        opt = {}
        for i in range(1, 10):
            opt[f"ref_image_{i}"] = ("IMAGE",)
        for i in range(1, 4):
            opt[f"ref_video_{i}"] = ("IMAGE", {"tooltip":
                "Video frames at 24 fps (same batch you feed the H3 node)."})
        return {
            "required": {
                "intent": ("STRING", {"multiline": True, "default":
                    "Use Picture 1 as the character identity. She walks "
                    "through a rainy neon alley at night.",
                    "tooltip": "What should happen, plus the JOB of every "
                    "connected asset (identity / style / motion / camera / "
                    "voice / edit source / BGM...). Include dialogue verbatim."}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600,
                                   "step": 17}),
                "video_1_has_audio": ("BOOLEAN", {"default": False,
                    "tooltip": "Set to match the ref_video_audio_1 connection "
                    "on the H3 node; shifts <Audio j> numbering."}),
                "video_2_has_audio": ("BOOLEAN", {"default": False}),
                "video_3_has_audio": ("BOOLEAN", {"default": False}),
                "standalone_audio_count": ("INT", {"default": 0, "min": 0,
                                                   "max": 3,
                    "tooltip": "How many standalone ref_audio inputs are "
                    "connected on the H3 node."}),
                "audio_notes": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "The writer cannot hear audio. Describe each "
                    "audio asset and its job, e.g. 'Audio 1: upbeat synth BGM "
                    "from Video 1, reuse fully. Audio 2: calm male voice, "
                    "timbre reference for the speaker.'"}),
                **BACKEND_INPUTS,
            },
            "optional": {
                "llm": ("H3_LLM", {"tooltip":
                    "Connect an H3 Prompt LLM Loader to run the writer "
                    "inside ComfyUI. Leave empty to use the HTTP fields."}),
                "instructions": ("H3_INSTRUCTIONS", {"tooltip":
                    "Optional H3 Prompt Instructions node to customize how "
                    "the LLM is guided. Empty = built-in defaults."}),
                **opt,
                **HTTP_INPUTS,
            },
        }

    def write(self, intent, length, video_1_has_audio, video_2_has_audio,
              video_3_has_audio, standalone_audio_count, audio_notes,
              temperature, seed, retry_on_invalid, llm=None, instructions=None,
              base_url="http://127.0.0.1:1234/v1", model="local",
              api_key="none", **refs):
        dur = duration_seconds(length)

        images = [(i, refs.get(f"ref_image_{i}")) for i in range(1, 10)]
        images = [(i, t) for i, t in images if t is not None]
        videos = [(i, refs.get(f"ref_video_{i}")) for i in range(1, 4)]
        videos = [(i, t) for i, t in videos if t is not None]
        has_audio = {1: video_1_has_audio, 2: video_2_has_audio,
                     3: video_3_has_audio}

        # Reproduce the H3 node's presentation order to derive fixed labels.
        asset_lines = []
        expected = []
        parts = []
        pic_n = vid_n = aud_n = 0
        for _, tensor in images:
            pic_n += 1
            label = f"<Picture {pic_n}>"
            expected.append(label)
            asset_lines.append(f"- {label}: reference image (attached below)")
            parts.append(client.text_part(f"This is {label}:"))
            parts.append(client.image_part(
                imaging.tensor_frame_to_b64(tensor[0])))
        for slot, tensor in videos:
            vid_n += 1
            if has_audio.get(slot):
                aud_n += 1
                expected.append(f"<Audio {aud_n}>")
                asset_lines.append(
                    f"- <Audio {aud_n}>: soundtrack of <Video {vid_n}> "
                    "(not audible to you; see the user's audio notes)")
            label = f"<Video {vid_n}>"
            expected.append(label)
            frames, stamps = imaging.sample_video_frames(tensor)
            secs = tensor.shape[0] / FPS
            asset_lines.append(
                f"- {label}: reference video, ~{secs:.1f}s "
                f"({len(frames)} sampled frames attached below)")
            parts.append(client.text_part(
                f"Sampled frames from {label} at "
                f"t={', '.join(f'{s:.2f}s' for s in stamps)}:"))
            parts.extend(client.image_part(f) for f in frames)
        for _ in range(standalone_audio_count):
            aud_n += 1
            expected.append(f"<Audio {aud_n}>")
            asset_lines.append(
                f"- <Audio {aud_n}>: standalone reference audio "
                "(not audible to you; see the user's audio notes)")

        header = (
            f"Video duration: {dur:.2f} seconds.\n"
            f"User intent:\n{intent.strip()}\n\n"
            "Connected reference assets (labels are FIXED by connection "
            "order; use them exactly):\n" + "\n".join(asset_lines))
        if audio_notes.strip():
            header += f"\n\nUser's audio notes:\n{audio_notes.strip()}"
        parts.insert(0, client.text_part(header))

        system = _system_for(guides.REF_SYSTEM, "override_ref", instructions)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": parts}]
        chat_fn = _make_chat_fn(llm, base_url, api_key, model,
                                temperature, seed)
        text, issues, attempts = _run_with_retry(
            messages, lambda t: validate.validate_ref(t, dur, expected),
            chat_fn, retry_on_invalid)
        return (text, _report("ref2va", dur, issues, attempts))


NODE_CLASS_MAPPINGS = {
    "H3PromptLLMLoader": H3PromptLLMLoader,
    "H3PromptInstructions": H3PromptInstructions,
    "H3PromptWriterKeyframes": H3PromptWriterKeyframes,
    "H3PromptWriterReference": H3PromptWriterReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PromptLLMLoader": "H3 Prompt LLM Loader (GGUF)",
    "H3PromptInstructions": "H3 Prompt Instructions",
    "H3PromptWriterKeyframes": "H3 Prompt Writer (T2V / I2V / FL2V)",
    "H3PromptWriterReference": "H3 Prompt Writer (Reference)",
}
