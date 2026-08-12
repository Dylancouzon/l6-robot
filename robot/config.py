"""Per-device calibration, read from `.env`.

Every number in this file depends on YOUR camera and YOUR scene, not on the
code. A 1080p webcam a foot from the desk and a 720p USB camera across the
room produce crops of different size and sharpness, and CLIP scores move with
them — so the recognition threshold that separates "this is my mug" from
"this is furniture" is not portable between machines. Calibrate it with
`testdata/verify_scores.py`; the README walks through reading its output.

Precedence, loosest to tightest: the defaults below, then `.env`, then a real
exported environment variable, then a command-line flag. That order is what
lets you keep a calibrated `.env` for the room and still A/B a value for one
run without editing a file.
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


# -- recognition -------------------------------------------------------------

# Nearest taught view must score at least this to count as "I know that".
# CLIP cosine has a high floor: two unrelated crops from the same camera
# routinely score 0.75-0.85, so this is a knee to find, not a fraction of
# confidence to guess at. Calibrate it, don't reason about it.
RECOGNIZE_THRESHOLD = _number("RECOGNIZE_THRESHOLD", 0.90)

# -- detection ---------------------------------------------------------------

# Detector confidence floor. Raise it to track less clutter.
DETECT_CONF = _number("DETECT_CONF", 0.30)

# Biggest proposal kept, as a fraction of frame area. The default drops
# torso-sized and wall-sized phantom boxes; raise it if your camera is far
# enough away that real objects are small in frame.
DETECT_MAX_AREA = _number("DETECT_MAX_AREA", 0.20)
