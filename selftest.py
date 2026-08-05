"""Offline self-test: validators vs the official MiniMax example prompts.

Run:  python selftest.py   (no ComfyUI, no LLM backend needed)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from h3_prompt_writer import validate
from h3_prompt_writer.nodes import align_frame_count, duration_seconds

I2VA_OFFICIAL = """For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the young woman shown in <Picture 1> remains beside the rain-covered train window, preserving her appearance, clothing, seat position, and the carriage layout. The camera trucks right with small amplitude at slow speed as she lifts her gaze from the folded letter toward the passing city lights. Her reflection moves across the glass while the quiet, breathy young woman (S1) says: <d>[English] I get off at the next station.</d> She folds the letter along its existing crease.

overall_soundscape: The train wheels produce a steady metallic rhythm beneath a low ventilation hum. Rain ticks against the window while paper rustles softly in her hands.

non_diegetic_music: Sustained cello notes at a slow tempo with widely spaced piano tones, gradually decreasing in volume."""

FL2VA_OFFICIAL = """How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 8.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a rain-soaked cyclist begins in the position and framing established by Picture 1, holding a closed black umbrella beside a silver bicycle. The camera pulls out with small amplitude at slow speed as she settles into the pose established by Picture 2 at the end of the shot.

overall_soundscape: Rain falls steadily on the pavement.

non_diegetic_music: N/A"""

REF_OFFICIAL = """subject_definitions:
<Subject 1> is the coffee-shop environment in <Picture 1>, featuring an exposed brick wall.
<Audio 1> is the voice-timbre reference for <Subject 3> (S1).

summary:
[reference generation + audio reference] The target video shows <Subject 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the environment is retained.
<Audio 1>: reference - its vocal timbre guides the dialogue delivery.

detailed_description:
The target video uses a realistic multi-camera sitcom style.
[Shot 1] A medium shot establishes <Subject 1>. The woman (S1) exclaims, <d>[English] Hey! Watch your dog!</d>
[Shot 2] At 00:03.000, the shot cuts to a close-up.

overall_soundscape:
Soft indoor coffee-shop room tone continues throughout the scene.

non_diegetic_music:
N/A"""

BAD_BASE = """A woman walks. [Pan left] She smiles.
overall_soundscape: wind."""


def check(name, issues, expect_ok):
    ok = (not issues) == expect_ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        for i in issues:
            print(f"      - {i}")
    return ok


def main():
    results = [
        check("official I2VA example validates",
              validate.validate_base(I2VA_OFFICIAL, "i2va", 8.0), True),
        check("official FL2VA example validates (8.00s)",
              validate.validate_base(FL2VA_OFFICIAL, "fl2va", 8.0), True),
        check("FL2VA duration mismatch is caught",
              validate.validate_base(FL2VA_OFFICIAL, "fl2va", 5.17), False),
        check("official ref example validates",
              validate.validate_ref(REF_OFFICIAL, 8.0,
                                    ["<Picture 1>", "<Audio 1>"]), True),
        check("missing label is caught",
              validate.validate_ref(REF_OFFICIAL, 8.0, ["<Video 1>"]), False),
        check("garbage base prompt is rejected",
              validate.validate_base(BAD_BASE, "i2va", 5.17), False),
        check("frame grid: 124 -> 5.17s",
              [] if f"{duration_seconds(124):.2f}" == "5.17" else ["bad"],
              True),
        check("frame grid: align 120 -> 124",
              [] if align_frame_count(120) == 124 else ["bad"], True),
    ]
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
