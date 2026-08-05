"""Smoke test for the in-process backend: load a VLM GGUF + mmproj and
run one tiny vision chat.

Set env vars before running (ComfyUI's python_embeded recommended):

  set H3_TEST_GGUF=C:\\path\\to\\model.gguf
  set H3_TEST_MMPROJ=C:\\path\\to\\mmproj.gguf
  python smoketest_local.py
"""

import base64
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image

from h3_prompt_writer import local_llm

model_path = os.environ.get("H3_TEST_GGUF", "").strip()
mmproj_path = os.environ.get("H3_TEST_MMPROJ", "").strip()
if not model_path or not os.path.isfile(model_path):
    sys.exit("Set H3_TEST_GGUF to an existing VLM .gguf path.")
if not mmproj_path or not os.path.isfile(mmproj_path):
    sys.exit("Set H3_TEST_MMPROJ to an existing mmproj .gguf path.")

handle = {
    "model_path": model_path,
    "mmproj_path": mmproj_path,
    "n_ctx": 8192,
    "n_gpu_layers": -1,
    "keep_loaded": False,
}

# solid red test image
buf = io.BytesIO()
Image.new("RGB", (256, 256), (220, 30, 30)).save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode()

messages = [
    {"role": "system", "content": "Answer in one short sentence."},
    {"role": "user", "content": [
        {"type": "text", "text": "What is the dominant color of this image?"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]},
]

t0 = time.time()
reply = local_llm.chat_local(handle, messages, temperature=0.1,
                             max_tokens=64, seed=42)
print(f"\n--- reply ({time.time() - t0:.1f}s total, incl. load) ---")
print(reply)
ok = "red" in reply.lower()
print("\nVISION:", "OK" if ok else "FAILED - model did not identify red")
sys.exit(0 if ok else 1)
