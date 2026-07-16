"""Scratch check: does a held-out view clear RECOGNIZE_THRESHOLD (0.80)
against taught views of the same object, and stay below it for a foreign
object? Run: uv run python testdata/verify_scores.py
"""
import itertools
from pathlib import Path

import cv2
import numpy as np
from fastembed import ImageEmbedding
from PIL import Image
from ultralytics import YOLO

TESTDATA = Path(__file__).parent
THRESHOLD = 0.80
PAD = 0.12

OBJECTS = {
    "rubberduck": ["rubberduck_1.jpg", "rubberduck_2.jpg", "rubberduck_3.jpg"],
    "vase": ["vase_1.jpg", "vase_2.jpg", "vase_3.jpg"],
}
HELD_OUT = {"rubberduck": "rubberduck_3.jpg", "vase": "vase_3.jpg"}
FOREIGN = "hardhat_1.jpg"


def padded_crop(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    return frame[y1:y2, x1:x2]


def detect_and_crop(model, path):
    frame = cv2.imread(str(path))
    results = model.predict(
        frame, conf=0.30, imgsz=640, agnostic_nms=True, verbose=False
    )[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        print(f"  ! no detection for {path.name}, using full image as crop")
        return frame
    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (
        boxes.xyxy[:, 3] - boxes.xyxy[:, 1]
    )
    box = boxes.xyxy[int(areas.argmax())].tolist()
    return padded_crop(frame, box)


def embed(embedder, crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return next(embedder.embed([pil]))


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    model = YOLO("yoloe-11l-seg-pf.pt")
    embedder = ImageEmbedding("Qdrant/clip-ViT-B-32-vision")

    all_files = sorted(set(itertools.chain(*OBJECTS.values())) | {FOREIGN})
    crops = {f: detect_and_crop(model, TESTDATA / f) for f in all_files}
    vecs = {f: embed(embedder, crops[f]) for f in all_files}

    print(f"\n{'pair':45s} {'cos sim':>8s}  clears 0.80?")
    print("-" * 70)
    for obj, files in OBJECTS.items():
        held_out = HELD_OUT[obj]
        taught = [f for f in files if f != held_out]
        for t in taught:
            sim = cosine(vecs[held_out], vecs[t])
            print(f"{held_out} vs {t:25s} {sim:8.3f}  "
                  f"{'YES' if sim >= THRESHOLD else 'no'}")
        for t in taught:
            sim = cosine(vecs[FOREIGN], vecs[t])
            print(f"{FOREIGN} vs {t:25s} {sim:8.3f}  "
                  f"{'YES (bad)' if sim >= THRESHOLD else 'no (good)'}")


if __name__ == "__main__":
    main()
