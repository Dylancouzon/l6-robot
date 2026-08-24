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


def _fractions(name, default):
    """A knob that takes one fraction for all four sides, or four of them as
    `left,top,right,bottom`. Fractions, not pixels, so it survives a change of
    capture size."""
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

# Smallest proposal kept, the other end of the same band. Drops speck noise —
# and, raised, the small far-away clutter that is technically an object but
# not what anyone is holding up to the camera. Area, not width: it scales as
# the SQUARE of how big the thing looks, so 0.001 is only about 12% wider than
# the 0.0008 this shipped with, not 25%.
DETECT_MIN_AREA = _number("DETECT_MIN_AREA", 0.001)

# -- camera ------------------------------------------------------------------

# How the camera is mounted, in degrees. The appliance's camera is fixed
# upside down in the chassis, so the unit's own .env says 180; a camera
# sitting the right way up wants 0, which is why that is the default here.
#
# Only 0 and 180 are accepted. A quarter turn would swap the frame's width and
# height, and the capture size, the page's video box and the scene pictures
# are all written for a landscape frame — that is a different job from a mount
# fix, so an unsupported value fails at startup instead of half-working.
CAMERA_ROTATE = _number("CAMERA_ROTATE", 0, int)
if CAMERA_ROTATE not in (0, 180):
    raise SystemExit(
        f"CAMERA_ROTATE in {ENV_FILE.name} must be 0 or 180, got "
        f"{CAMERA_ROTATE}. A quarter turn would swap the frame's width and "
        "height, which the rest of the app assumes is landscape.")


# What to cut off each edge of the camera frame, as fractions of the width and
# height: one number for all four sides, or `left,top,right,bottom`.
#
# A lens whose image circle is smaller than the sensor puts its own black rim
# in the picture, and the detector proposes boxes on that rim like any other
# shape. Measure it rather than guessing: hold a white card against the lens,
# capture one frame, and the rim is everything that stays black.
#
# Four numbers and not one because the rim is rarely centred. On the appliance
# the circle sits 104 px low and 23 px left of the sensor centre, so one
# symmetric fraction big enough to clear the worst edge keeps 50% of the
# picture where four edge fractions keep 62%.
#
# Read in the picture's own coordinates: the mount rotation above happens
# first, so `top` is the top of the feed you are looking at. Changing the
# mount means measuring these again.
FRAME_CROP = _fractions("FRAME_CROP", (0.0, 0.0, 0.0, 0.0))
if FRAME_CROP[0] + FRAME_CROP[2] > 0.8 or FRAME_CROP[1] + FRAME_CROP[3] > 0.8:
    raise SystemExit(
        f"FRAME_CROP in {ENV_FILE.name} cuts away almost the whole frame: "
        f"{FRAME_CROP}")
