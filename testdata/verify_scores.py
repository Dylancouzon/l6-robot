"""Calibrate RECOGNIZE_THRESHOLD for YOUR camera, and check it still holds.

Run: uv run python testdata/verify_scores.py [--source DIR]

Two jobs:

1. Separation. A held-out view of an object must out-score every foreign
   object by a clear margin. If it doesn't, no threshold can work and the
   problem is the crops, not the number.
2. The knee. It sweeps candidate thresholds and prints where same-object
   matches survive while cross-object matches die.

It embeds crops through `robot.brain.detect.padded_crop` with the segmentation
mask, the same call the live robot makes, because the crop pipeline moves the
scores as much as the encoder does: a plain crop of a studio photo and a
masked crop of a small live detection are different distributions, and a
threshold tuned on one does not transfer to the other.

It is not the whole live path: it takes the largest proposal, where the robot
applies an area band and picks by salience, then keeps the sharpest crop over
several frames. The mask and the encoder are shared, which is what moves the
scores; frame selection doesn't.

Point --source at a directory of your own photos to calibrate for your
camera; name the files `<object>_<n>.jpg` so views of one object group.
"""
import argparse
import itertools
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import robot/

import cv2
import numpy as np
from fastembed import ImageEmbedding
from PIL import Image
from ultralytics import YOLO

from robot.config import RECOGNIZE_THRESHOLD
from robot.brain.detect import CONF, IMGSZ, padded_crop
from robot.brain.models import CACHE_DIR, CLIP_VISION_MODEL, ENCODER_THREADS

TESTDATA = Path(__file__).parent
SWEEP = [0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]


def group_of(path):
    """`rubberduck_2.jpg` -> `rubberduck`: views of one object share a stem."""
    return re.sub(r"[_-]?\d+$", "", path.stem) or path.stem


def detect_and_crop(model, path):
    """The biggest proposal, cropped exactly as the live robot crops it."""
    frame = cv2.imread(str(path))
    if frame is None:
        return None
    results = model.predict(frame, conf=CONF, imgsz=IMGSZ,
                            agnostic_nms=True, verbose=False)[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        print(f"  ! no detection for {path.name}, using the full image")
        return frame
    areas = ((boxes.xyxy[:, 2] - boxes.xyxy[:, 0])
             * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1]))
    i = int(areas.argmax())
    polys = results.masks.xy if results.masks is not None else None
    mask = polys[i] if polys is not None and i < len(polys) else None
    return padded_crop(frame, boxes.xyxy[i].tolist(), mask)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(TESTDATA),
                    help="directory of <object>_<n>.jpg views (default: testdata/)")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"--source is not a directory: {source}")
    paths = sorted(p for p in source.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(paths) < 2:
        raise SystemExit(f"need at least 2 images in {source}")

    model = YOLO("yoloe-11l-seg-pf.pt")
    # cache_dir, like the app: without it FastEmbed re-downloads 336 MB into
    # /tmp, which is both slow and a copy that /tmp's cleaner will delete.
    embedder = ImageEmbedding(CLIP_VISION_MODEL, threads=ENCODER_THREADS,
                              cache_dir=CACHE_DIR)

    crops, kept = {}, []
    for p in paths:
        crop = detect_and_crop(model, p)
        if crop is not None and crop.size:
            crops[p] = crop
            kept.append(p)
    vecs = {}
    embedded = embedder.embed([Image.fromarray(crops[p][:, :, ::-1])
                               for p in kept])
    for p, v in zip(kept, embedded):
        vecs[p] = np.asarray(v) / np.linalg.norm(v)

    groups = defaultdict(list)
    for p in kept:
        groups[group_of(p)].append(p)

    same, cross = [], []
    print(f"\n{'pair':52s} {'cos sim':>8s}")
    print("-" * 64)
    for a, b in itertools.combinations(kept, 2):
        sim = float(vecs[a] @ vecs[b])
        pair = f"{a.name} vs {b.name}"
        if group_of(a) == group_of(b):
            same.append(sim)
            print(f"{pair:52s} {sim:8.3f}  same object")
        else:
            cross.append(sim)
    for a, b in itertools.combinations(kept, 2):
        if group_of(a) != group_of(b):
            print(f"{a.name + ' vs ' + b.name:52s} "
                  f"{float(vecs[a] @ vecs[b]):8.3f}  different")

    if not same or not cross:
        raise SystemExit(
            "\nneed at least two objects, and at least two views of one of "
            "them, to calibrate. Name files <object>_<n>.jpg.")

    same, cross = np.array(same), np.array(cross)
    print(f"\nsame object   min={same.min():.3f} med={np.median(same):.3f}")
    print(f"different     med={np.median(cross):.3f} max={cross.max():.3f}")
    margin = same.min() - cross.max()
    print(f"margin (worst same - best different): {margin:+.3f}")
    if margin <= 0:
        print("  ! the distributions overlap. No threshold separates them — "
              "fix the crops (get closer, better light) before tuning.")

    sweep = sorted(set(SWEEP + [RECOGNIZE_THRESHOLD]))
    print(f"\n{'threshold':>9s} {'same kept':>10s} {'different wrongly matched':>26s}")
    for th in sweep:
        mark = "  <- .env RECOGNIZE_THRESHOLD" if th == RECOGNIZE_THRESHOLD else ""
        print(f"{th:9.2f} {f'{(same >= th).sum()}/{len(same)}':>10s} "
              f"{f'{(cross >= th).sum()}/{len(cross)}':>26s}{mark}")

    good = [th for th in sweep if (cross >= th).sum() == 0
            and (same >= th).sum() == len(same)]
    print(f"\nclean on these images: "
          f"{good if good else 'nothing — fix the crops before tuning'}")

    # The instruction has to be "start at the top", not "pick the middle".
    # Separate photos of distinct objects are the EASY case: they share no
    # background, lighting or scale, so cross-object scores here sit far below
    # what a live scene produces. This range is a floor, and a threshold
    # chosen from the middle of it will match half the room on real video.
    if good:
        print(f"\nStart at the TOP of that range ({max(good):.2f}), not the "
              f"middle, and put it in .env as RECOGNIZE_THRESHOLD.")
    print("Then check it against live video, which is the case that decides "
          "it: separate\nphotos of distinct objects share no background, "
          "lighting or scale, so they score\nfar apart here and far closer "
          "in a real scene. This range is a FLOOR.")
    if Path(args.source).resolve() == TESTDATA.resolve():
        print(f"\nThese are the bundled demo photos, not your camera. The "
              f"shipped default\n({RECOGNIZE_THRESHOLD}) is deliberately "
              f"above what they can justify, because it was\ncalibrated on "
              f"live scene crops. Re-run with --source on your own photos.")


if __name__ == "__main__":
    main()
