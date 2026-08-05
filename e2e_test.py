"""End-to-end test: run the keyframes writer with a local GGUF VLM.

Builds a synthetic first frame, runs H3PromptWriterKeyframes, prints the
generated I2VA prompt and validation report.

  set H3_TEST_GGUF=C:\\path\\to\\model.gguf
  set H3_TEST_MMPROJ=C:\\path\\to\\mmproj.gguf
  python e2e_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch

from h3_prompt_writer.nodes import H3PromptWriterKeyframes

model_path = os.environ.get("H3_TEST_GGUF", "").strip()
mmproj_path = os.environ.get("H3_TEST_MMPROJ", "").strip()
if not model_path or not os.path.isfile(model_path):
    sys.exit("Set H3_TEST_GGUF to an existing VLM .gguf path.")
if not mmproj_path or not os.path.isfile(mmproj_path):
    sys.exit("Set H3_TEST_MMPROJ to an existing mmproj .gguf path.")

llm = {
    "model_path": model_path,
    "mmproj_path": mmproj_path,
    "n_ctx": 16384,
    "n_gpu_layers": -1,
    "keep_loaded": False,
}

# synthetic first frame: warm gradient "sunset" with a dark silhouette bar
h, w = 512, 768
y = torch.linspace(0.9, 0.2, h).unsqueeze(1).expand(h, w)
img = torch.stack([y, y * 0.5, y * 0.25], dim=-1)  # orange gradient
img[380:512, :, :] = 0.06  # dark foreground
frame = img.unsqueeze(0)  # [1, H, W, C]

node = H3PromptWriterKeyframes()
prompt, report = node.write(
    intent=("The camera slowly pushes in toward the horizon as the sky "
            "shifts and glows. Gentle wind. No dialogue, soft ambient "
            "music."),
    length=124,
    temperature=0.4,
    seed=7,
    retry_on_invalid=True,
    llm=llm,
    first_frame=frame,
)

print("=" * 70)
print(prompt)
print("=" * 70)
print(report)
sys.exit(0 if "validation: OK" in report else 1)
