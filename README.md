# ComfyUI-H3-PromptWriter

LLM/VLM-backed prompt writers for the open-weight **MiniMax H3** video model —
a local stand-in for the closed **H3-Context-IR** module. You write casual
intent; the node writes the structured prompt H3-Base actually expects and
feeds it to the native `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo`
nodes.

Prompt formats follow the official guides in
[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
(`docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md` and `..._ref_en.md`).

## Install

1. Copy this folder into `ComfyUI/custom_nodes/ComfyUI-H3-PromptWriter`
2. Restart ComfyUI
3. Nodes appear under **MiniMax-H3/prompting**

No `pip install` is required for the **HTTP / OpenAI** path (recommended when
sharing). Local GGUF needs an extra package — see below.

To use the generated prompts you also need ComfyUI **0.30.0+** (native MiniMax
H3 nodes) and the H3 model files from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3). Those are
separate from this custom node package.

## Dependencies

### HTTP / OpenAI / OpenRouter (recommended share path)

| Dependency | Notes |
|---|---|
| ComfyUI stock packages | Uses `requests`, `numpy`, `PIL` already in ComfyUI |
| API key | Your own key — never ship one in a workflow |

Leave the **H3 Prompt LLM Loader** unconnected. On the writer node set:

| Field | OpenAI | OpenRouter |
|---|---|---|
| `base_url` | `https://api.openai.com/v1` | `https://openrouter.ai/api/v1` |
| `model` | `gpt-4o` / `gpt-4.1` (needs vision for I2V/R2V) | e.g. `openai/gpt-4o` |
| `api_key` | `sk-...` | your OpenRouter key |

Same fields also work with LM Studio (`http://127.0.0.1:1234/v1`), llama.cpp
server (`:8080/v1`), Ollama (`:11434/v1`), etc.

### Local GGUF (optional, advanced)

Only needed if you connect **H3 Prompt LLM Loader**. Extra dependency:

- **`llama-cpp-python`** with a llama.cpp build new enough for your model
  (Qwen3-VL needs a late-2025+ build with `Qwen3VLChatHandler` or equivalent)

This is the painful part when sharing. The wheel must match:

- Python version (e.g. 3.13 → `cp313`)
- CUDA version (e.g. 13.0 → `cu130`)
- GPU architecture
- CPU instruction set — many prebuilt wheels are compiled with **AVX-512** and
  crash with `OSError: [WinError -1073741795] / 0xc000001d` on AVX2-only CPUs
  (e.g. Ryzen 5000). Prefer wheels with **runtime CPU dispatch** (multiple
  `ggml-cpu-*.dll` variants), such as
  [JamePeng/llama-cpp-python releases](https://github.com/JamePeng/llama-cpp-python/releases).

`pip install llama-cpp-python` from PyPI alone is often wrong for Windows + CUDA
+ modern VLMs. Install a matching wheel for *your* machine, then restart ComfyUI.

Also place in `ComfyUI/models/llm_gguf/` (or `models/LLM/`):

- A VLM **GGUF** (e.g. Qwen3-VL-4B)
- Its matching **mmproj** (required so the writers can see reference images)

Text-only GGUFs work for T2V (no images) only.

VRAM tip: leave `keep_loaded` **off** so the VLM unloads after writing the
prompt and H3 can use the GPU. Turning it on needs enough VRAM for both.

`llama_cpp` is imported only when the local loader actually runs — a missing
install does **not** break the HTTP / OpenAI path.

## Sharing this package

Zip the `ComfyUI-H3-PromptWriter` folder and send it. For most people, ship a
workflow that uses **HTTP OpenAI** (no loader connected) and tell them to paste
their own API key.

| Path | Pain level | What they need |
|---|---|---|
| HTTP / OpenAI | Easy | This folder + Comfy 0.30+ + H3 models + API key |
| Local GGUF | High | Everything above + correct `llama-cpp-python` wheel + GGUF + mmproj |

## Nodes

### H3 Prompt LLM Loader (GGUF)

Runs the writer LLM **inside the ComfyUI process** — no external server.
Pick a GGUF from `models/llm_gguf` (or `models/LLM`) and its matching
mmproj (required for the writers to see reference images). Loading is lazy;
with `keep_loaded` off, the model is freed right after each prompt so VRAM
goes back to H3 sampling. Wire its `llm` output into either writer node.

If `llm` is connected, it takes priority over the writer's HTTP fields.

### H3 Prompt Instructions

Optional node to customize how the writer LLM is guided. Leave it
disconnected and the writers use the built-in guides exactly as before.

- `extra_instructions` — appended to the built-in guide as **session
  directives that take priority over the defaults**: tone, style
  conventions, content-handling rules, vocabulary, pacing, etc. Applies to
  both writers.
- `override_keyframes_guide` / `override_reference_guide` — advanced:
  **replaces** the entire built-in system guide for the matching writer.
- `default_keyframes_guide` / `default_reference_guide` outputs expose the
  built-in guide texts. Wire one to a Show Text / Show Any node, copy it,
  edit, and paste it into the override field to start from the defaults.

Note: the format validator still runs on the output. If you override a
guide, keep the structural requirements (alignment lines and core fields
for keyframes; the six sections and fixed asset labels for reference) or
every run will end in validation issues.

Wire the `instructions` output into the `instructions` input of either (or
both) writer nodes.

### H3 Prompt Writer (T2V / I2V / FL2V)

For the **FL2VA** checkpoint. Mode is auto-detected from connections:

| Connected | Mode | Output starts with |
|---|---|---|
| nothing | t2va | the three core fields |
| `first_frame` | i2va | the exact I2VA alignment line |
| `first_frame` + `last_frame` | fl2va | two-picture alignment line, last mark = duration |
| `last_frame` only | l2va | last-frame alignment line |

Wire `prompt` → `MiniMaxH3ImageToVideo.prompt`, and use the **same `length`
value** on both nodes (frames at 24 fps on the 17k+5 grid; 124 ≈ 5.17 s —
the alignment timestamps depend on it).

### H3 Prompt Writer (Reference)

For the **Ref2VA** checkpoint. Connect the same images/video frame batches
you feed `MiniMaxH3ReferenceToVideo`, in the same slot order, so the
`<Picture i> / <Video k> / <Audio j>` labels line up. Set the
`video_N_has_audio` flags and `standalone_audio_count` to match your audio
connections — audio label numbering shifts with them (a video's soundtrack
label precedes the video; standalone audio comes last).

The writer cannot hear audio: describe each audio asset and its job in
`audio_notes`.

Output is the full six-section format (`subject_definitions`, `summary` with
task types, `retention_analysis` with retention markers,
`detailed_description`, `overall_soundscape`, `non_diegetic_music`).

## Validation

Every generated prompt is checked deterministically (required sections and
order, exact alignment lines, duration marks, shot timestamps inside the
video, balanced `<d>` tags, no old-style `[Pan left]` bracket commands,
known task types and retention markers, all connected labels mentioned).
On failure the issues are sent back to the LLM for one correction pass.
The `report` output shows mode, duration, and any remaining issues — wire it
to a text preview node.

## Offline self-test

```
python selftest.py
```

Validates the checkers against MiniMax's official example prompts. Does not
need ComfyUI, an API key, or `llama-cpp-python`.

## Tips

- Put dialogue lines you want spoken in the intent **verbatim, in quotes** —
  the writer must preserve them exactly inside `<d>` tags.
- Temperature 0.3–0.5 works well; raise it if outputs feel templated.
- Change `seed` to force a re-run (ComfyUI caches node outputs otherwise).
- For the Reference node, tell the intent what each asset is FOR
  ("Picture 1 = identity, Video 1 = camera move only") — explicit job
  assignment is the single highest-leverage habit with Ref2VA.
- Use a **vision** model for I2V/R2V (images are sent as base64). Text-only
  models are fine for T2V with no frames connected.
- If the writer refuses or rewrites your intent, that behavior comes from
  the LLM itself, not from these nodes. Steering via **H3 Prompt
  Instructions** helps, but the model's own alignment ultimately decides —
  choose a writer model whose policies fit your use case.
