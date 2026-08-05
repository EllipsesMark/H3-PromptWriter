"""Deterministic format validation for generated H3 prompts."""

import re

BASE_FIELDS = ["integrated_multimodal_description:", "overall_soundscape:",
               "non_diegetic_music:"]
REF_SECTIONS = ["subject_definitions:", "summary:", "retention_analysis:",
                "detailed_description:", "overall_soundscape:",
                "non_diegetic_music:"]
TASK_TYPES = {"keyframe completion", "reference generation", "video editing",
              "video continuation", "audio reuse", "audio reference"}
VISUAL_MARKERS = {"fully_preserved", "partially_preserved",
                  "attribute_transfer", "weak_reference"}
AUDIO_MARKERS = {"fully_copy", "partially_copy", "reference", "weak_reference"}

I2VA_LINE = ("For the target video, at 0.00 seconds into the target video, "
             "<Picture 1> (from [Shot 1]) is fully referenced.")


def strip_fences(text):
    """LLMs love wrapping output in ``` fences; remove them if present."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def _check_shared(text, duration, issues):
    if text.count("<d>") != text.count("</d>"):
        issues.append("Unbalanced <d>/</d> dialogue tags.")
    for m in re.finditer(r"At (\d{2}):(\d{2})\.(\d{3})", text):
        t = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1000.0
        if t >= duration:
            issues.append(
                f"Shot cut time {m.group(0)} is not within the "
                f"{duration:.2f}s video duration.")
    if re.search(r"\[(?:Push in|Pull out|Pan left|Pan right|Zoom in|Zoom out|"
                 r"Truck left|Truck right|Tilt up|Tilt down|Pedestal up|"
                 r"Pedestal down|Static shot|Tracking shot|Shake)[^\]]*\]",
                 text, re.IGNORECASE):
        issues.append("Bracketed camera commands found; camera motion must be "
                      "written as natural prose (that syntax is for the old "
                      "Hailuo API, not H3).")


def _fields_in_order(text, fields, issues):
    pos = -1
    for f in fields:
        p = text.find("\n" + f) if not text.startswith(f) else 0
        if p == -1 and not text.startswith(f):
            issues.append(f"Missing required section `{f}`.")
            continue
        p = text.find(f)
        if p < pos:
            issues.append(f"Section `{f}` is out of order.")
        pos = p


def validate_base(text, mode, duration):
    """mode: t2va | i2va | fl2va | l2va. Returns list of issue strings."""
    issues = []
    dur = f"{duration:.2f}"
    first_line = text.split("\n", 1)[0].strip()

    if mode == "t2va":
        if "reference pictures" in first_line or first_line.startswith(
                "For the target video"):
            issues.append("T2VA must not contain an image-alignment "
                          "instruction line.")
        if not first_line.startswith("integrated_multimodal_description:"):
            issues.append("T2VA must begin directly with "
                          "`integrated_multimodal_description:`.")
    elif mode == "i2va":
        if first_line != I2VA_LINE:
            issues.append("First line must be exactly: " + I2VA_LINE)
    elif mode == "fl2va":
        if not first_line.startswith(
                "How the reference pictures align with the target video"):
            issues.append("FL2VA must begin with the two-picture alignment "
                          "instruction line.")
        if f"{dur}-second mark" not in first_line:
            issues.append(f"Picture 2 must align with the {dur}-second mark "
                          "(the exact video duration, two decimals).")
        if "0.00-second mark" not in first_line:
            issues.append("Picture 1 must align with the 0.00-second mark.")
    elif mode == "l2va":
        if not first_line.startswith(
                "How the reference pictures align with the target video"):
            issues.append("L2VA must begin with the last-frame alignment "
                          "instruction line.")
        if f"{dur}-second mark" not in first_line:
            issues.append(f"<Picture 1> must align with the {dur}-second "
                          "mark (the exact video duration, two decimals).")

    _fields_in_order(text, BASE_FIELDS, issues)
    if "[Shot 1]" not in text:
        issues.append("Missing `[Shot 1]` opening shot marker.")
    _check_shared(text, duration, issues)
    return issues


def validate_ref(text, duration, expected_labels=()):
    """expected_labels: iterable like ('<Picture 1>', '<Video 1>', ...)."""
    issues = []
    _fields_in_order(text, REF_SECTIONS, issues)

    m = re.search(r"^summary:\s*\n?\s*\[([^\]]+)\]", text, re.MULTILINE)
    if not m:
        issues.append("`summary:` must start with a bracketed task-type "
                      "prefix like `[reference generation]`.")
    else:
        for t in (s.strip() for s in m.group(1).split("+")):
            if t not in TASK_TYPES:
                issues.append(f"Unknown task type `{t}` in summary prefix. "
                              f"Allowed: {sorted(TASK_TYPES)}.")

    ra = re.search(r"retention_analysis:\s*\n(.*?)(?=\ndetailed_description:)",
                   text, re.DOTALL)
    if ra:
        for line in filter(None, (l.strip() for l in ra.group(1).splitlines())):
            marker = re.search(r":\s*([a-z_]+)\s*-", line)
            if marker and marker.group(1) not in VISUAL_MARKERS | AUDIO_MARKERS:
                issues.append(f"Unknown retention marker `{marker.group(1)}` "
                              f"in line: {line[:80]}")

    for label in expected_labels:
        if label not in text:
            issues.append(f"Connected reference {label} is never mentioned "
                          "in the prompt.")

    if "[Shot 1]" not in text:
        issues.append("Missing `[Shot 1]` opening shot marker.")
    _check_shared(text, duration, issues)
    return issues
