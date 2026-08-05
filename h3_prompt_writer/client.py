"""Minimal OpenAI-compatible chat client.

Works with llama.cpp server, LM Studio, koboldcpp, Ollama (/v1), vLLM,
OpenRouter, OpenAI, etc. Images are sent as data-URI image_url parts,
which is the de-facto standard all of the above accept for VLMs.
"""

import json

import requests


class ChatClientError(RuntimeError):
    pass


def chat(base_url, api_key, model, messages, temperature=0.4,
         max_tokens=3072, seed=-1, timeout=300):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.lower() not in ("", "none"):
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed >= 0:
        payload["seed"] = seed

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload),
                             timeout=timeout)
    except requests.RequestException as e:
        raise ChatClientError(
            f"Could not reach LLM backend at {url}: {e}") from e

    if resp.status_code != 200:
        raise ChatClientError(
            f"LLM backend returned HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ChatClientError(
            f"Unexpected response shape from LLM backend: {str(data)[:500]}") from e

    # Some backends return content as a list of parts.
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") for p in content if isinstance(p, dict))
    return (content or "").strip()


def text_part(text):
    return {"type": "text", "text": text}


def image_part(b64_jpeg):
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"}}
