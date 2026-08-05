"""In-process GGUF inference via llama-cpp-python.

Loads the model inside the ComfyUI process (no external server). One model is
kept resident at a time (single-slot cache); it can be freed after each write
so VRAM goes back to H3 sampling.
"""

import gc

_CACHE = {"key": None, "llama": None, "handler": None}


class LocalLLMError(RuntimeError):
    pass


def unload():
    if _CACHE["llama"] is not None:
        try:
            _CACHE["llama"].close()
        except Exception:
            pass
    _CACHE.update({"key": None, "llama": None, "handler": None})
    gc.collect()


def _load(model_path, mmproj_path, n_ctx, n_gpu_layers):
    key = (model_path, mmproj_path, n_ctx, n_gpu_layers)
    if _CACHE["key"] == key and _CACHE["llama"] is not None:
        return _CACHE["llama"]
    unload()

    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise LocalLLMError(
            "llama-cpp-python is not installed in this ComfyUI environment. "
            "Install it, or use the HTTP backend fields instead.") from e

    handler = None
    if mmproj_path:
        import llama_cpp.llama_chat_format as chat_format
        # Prefer the Qwen3-VL handler when this build has it.
        cls = getattr(chat_format, "Qwen3VLChatHandler", None) or \
            getattr(chat_format, "Qwen25VLChatHandler")
        handler = cls(clip_model_path=mmproj_path, verbose=False)

    llama = Llama(
        model_path=model_path,
        chat_handler=handler,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    _CACHE.update({"key": key, "llama": llama, "handler": handler})
    return llama


def chat_local(handle, messages, temperature=0.4, max_tokens=3072, seed=-1):
    """handle: dict from the loader node. messages: OpenAI-style messages
    (image_url data-URI parts supported when an mmproj is loaded)."""
    has_images = any(
        isinstance(m.get("content"), list)
        and any(p.get("type") == "image_url" for p in m["content"])
        for m in messages)
    if has_images and not handle.get("mmproj_path"):
        raise LocalLLMError(
            "This prompt includes reference images but the loaded GGUF has no "
            "mmproj (vision projector). Pick a VLM + its mmproj in the "
            "H3 Prompt LLM Loader.")

    llama = _load(handle["model_path"], handle.get("mmproj_path"),
                  handle.get("n_ctx", 16384), handle.get("n_gpu_layers", -1))
    kwargs = {"messages": messages, "temperature": temperature,
              "max_tokens": max_tokens}
    if seed >= 0:
        kwargs["seed"] = seed
    try:
        out = llama.create_chat_completion(**kwargs)
    finally:
        if not handle.get("keep_loaded", False):
            unload()
    content = out["choices"][0]["message"]["content"] or ""
    return content.strip()
