"""Load camera and microphone settings from `.env`.

Camera-dependent values should be calibrated with `testdata/verify_scores.py`.
Exported environment variables override `.env`; command-line flags override
both when available.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# override=False is the important half: an exported variable beats the file.
load_dotenv(ENV_FILE, override=False)


def _number(name, default, cast=float):
    """One calibration knob, with a legible error instead of a stack trace."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise SystemExit(
            f"{name} in {ENV_FILE.name} must be a number, got {raw!r}")


def _fractions(name, default):
    """Read one edge fraction or `left,top,right,bottom` fractions."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) == 1:
        parts *= 4
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        raise SystemExit(
            f"{name} in {ENV_FILE.name} must be one number or four "
            f"(left,top,right,bottom), got {raw!r}")
    if len(vals) != 4 or not all(0 <= v < 0.5 for v in vals):
        raise SystemExit(
            f"{name} in {ENV_FILE.name} wants four fractions of 0 to 0.5 as "
            f"left,top,right,bottom, got {raw!r}")
    return vals


def _device(name):
    """An optional sounddevice name or numeric index."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


# -- recognition -------------------------------------------------------------

# The nearest taught view must meet this CLIP cosine-similarity threshold.
# It is not a confidence percentage; calibrate it for each camera.
RECOGNIZE_THRESHOLD = _number("RECOGNIZE_THRESHOLD", 0.90)

# -- detection ---------------------------------------------------------------

# Detector confidence floor. Raise it to track less clutter.
DETECT_CONF = _number("DETECT_CONF", 0.30)

# Largest detection kept, as a fraction of the frame area.
DETECT_MAX_AREA = _number("DETECT_MAX_AREA", 0.20)

# Smallest detection kept, as a fraction of the frame area.
DETECT_MIN_AREA = _number("DETECT_MIN_AREA", 0.001)

# -- camera ------------------------------------------------------------------

# The appliance camera is mounted upside down. Other cameras normally use 0.
# Quarter turns are unsupported because the interface expects landscape video.
CAMERA_ROTATE = _number("CAMERA_ROTATE", 0, int)
if CAMERA_ROTATE not in (0, 180):
    raise SystemExit(
        f"CAMERA_ROTATE in {ENV_FILE.name} must be 0 or 180, got "
        f"{CAMERA_ROTATE}. A quarter turn would swap the frame's width and "
        "height, which the rest of the app assumes is landscape.")


# Fractions removed from the camera edges to hide a lens rim. Accepts one value
# for every edge or `left,top,right,bottom`. Rotation happens before cropping,
# so the directions match the displayed image.
FRAME_CROP = _fractions("FRAME_CROP", (0.0, 0.0, 0.0, 0.0))
if FRAME_CROP[0] + FRAME_CROP[2] > 0.8 or FRAME_CROP[1] + FRAME_CROP[3] > 0.8:
    raise SystemExit(
        f"FRAME_CROP in {ENV_FILE.name} cuts away almost the whole frame: "
        f"{FRAME_CROP}")

# -- microphone --------------------------------------------------------------

MIC_DEVICE = _device("MIC_DEVICE")
