"""Tensor -> base64 JPEG helpers for feeding ComfyUI IMAGE inputs to a VLM."""

import base64
import io

import numpy as np
from PIL import Image

MAX_SIDE = 1024  # keep VLM token cost sane; identity detail survives fine


def tensor_frame_to_b64(frame, max_side=MAX_SIDE):
    """frame: [H, W, C] float tensor in 0..1 -> base64 JPEG string."""
    arr = (frame.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(arr[..., :3])
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * scale)),
                          max(1, round(img.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def sample_video_frames(batch, max_frames=4):
    """batch: [B, H, W, C] IMAGE tensor treated as 24fps video frames.

    Returns (b64_list, timestamps_s) evenly sampled across the clip.
    """
    n = batch.shape[0]
    if n <= max_frames:
        idx = list(range(n))
    else:
        step = (n - 1) / (max_frames - 1)
        idx = sorted({round(i * step) for i in range(max_frames)})
    return ([tensor_frame_to_b64(batch[i]) for i in idx],
            [round(i / 24.0, 2) for i in idx])
