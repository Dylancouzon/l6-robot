"""Detection and cadence — the two "new pieces" L6 names but doesn't teach.

YOLOE (prompt-free) finds and crops objects; its labels are discarded —
detection finds *a thing*, memory tells it *which* thing. Class names are
consulted only to suppress people/hands/faces, and that suppression is
sticky per track because classifiers flicker.

Cadence: a track must be stable (N consecutive frames) before it becomes a
match candidate or teachable, and it re-queries memory every couple of
seconds, not per frame. Design and all tuning constants lifted from
qdrant-labs/memory-fleet, where they were field-tuned on live desk scenes.
"""
import os
import time

import cv2
import numpy as np

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # no pip calls at runtime

try:
    # torch-MPS autoreleases Metal objects per inference; a pure-Python loop
    # never drains the pool, leaking ~80 MB/min. Every model call must run
    # inside this pool so the objects are released each iteration.
    from objc import autorelease_pool
except ImportError:  # non-macOS: nothing to drain
    from contextlib import nullcontext as autorelease_pool

CONF = 0.30        # calibration knob (--conf)
IMGSZ = 640
MAX_DET = 64
# Normalized box-area band: drops speck noise AND oversized phantom/torso
# regions. Demo objects are hand-held scale, so the cap stays tight.
MIN_AREA = 0.0008
MAX_AREA = 0.20    # calibration knob (--max-area)
STABLE_FRAMES = 3
REQUERY_SECONDS = 2.0
DEAD_SECONDS = 1.5
PAD = 0.12         # small margin; the mask removes the background anyway
FILL = (124, 124, 124)

# People, faces, and body parts are never objects to remember. Word list and
# matching copied verbatim from memory-fleet's detector (field-tuned there).
PERSON_WORDS = frozenset(
    "person people man men woman women boy girl child kid baby human humans face "
    "faces head hair ear eye eyes nose mouth lip lips chin cheek forehead beard "
    "mustache moustache neck shoulder arm arms elbow wrist hand hands finger "
    "fingers thumb fist chest torso waist hip leg legs knee ankle foot feet toe "
    "toes skin body "
    "wig ponytail braid bangs afro dreadlock dreadlocks mane haircut hairstyle "
    "eyebrow eyebrows eyelash eyelashes lash lashes freckle freckles jaw scalp "
    "sideburn sideburns goatee tongue tooth teeth throat nostril manicure "
    "businessman fisherman fireman airman craftsman".split()
)


def is_person_like(class_name):
    return any(w in PERSON_WORDS
               for w in class_name.lower().replace("-", " ").split())


def padded_crop(frame, box, mask=None):
    """Crop with 12% padding; flatten the background to neutral gray inside
    the segmentation mask. The mask fill is what makes recognition survive
    background and hand changes (memory-fleet's crops.py, verbatim logic)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    region = frame[y1:y2, x1:x2]
    if mask is not None and len(mask) >= 3 and region.size:
        full = np.zeros((h, w), np.uint8)
        cv2.fillPoly(full, [np.asarray(mask, dtype=np.int32)], 255)
        full = cv2.dilate(full, np.ones((7, 7), np.uint8), iterations=1)
        inside = full[y1:y2, x1:x2] > 0
        region = region.copy()
        region[~inside] = FILL
    return region


def crop_quality(frame, box, conf):
    """Bigger, sharper, more confident crops make better object portraits."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    small = cv2.resize(frame[y1:y2, x1:x2], (96, 96),
                       interpolation=cv2.INTER_AREA)
    sharp = cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                          cv2.CV_32F).var()
    area = (x2 - x1) * (y2 - y1) / (w * h)
    return conf * (area ** 0.5) * min(sharp, 1200.0)


class Track:
    def __init__(self, tid):
        self.tid = tid
        self.frames = 0
        self.last_seen = 0.0
        self.last_query = 0.0
        self.box = None
        self.crop = None       # best (sharpest) padded crop since last query
        self.crop_q = 0.0
        self.salience = 0.0    # size x centrality: what the robot attends to
        self.label = None      # from memory, never from the detector
        self.note = None       # the taught transcript, recalled on match
        self.score = 0.0
        self.vec = None        # last CLIP embedding of the crop
        self.sighted = False   # one "seen" memory per track, not per frame

    @property
    def stable(self):
        return self.frames >= STABLE_FRAMES

    def due_for_query(self, now):
        return self.stable and now - self.last_query >= REQUERY_SECONDS


class Detector:
    """YOLOE + BoT-SORT tracking + the stability gate."""

    def __init__(self, weights="yoloe-11l-seg-pf.pt", conf=CONF,
                 max_area=MAX_AREA):
        from pathlib import Path
        from ultralytics import YOLO
        import torch
        self.conf = conf
        self.max_area = max_area
        # resolve against the repo root, not the cwd — otherwise running from
        # another directory silently re-downloads the 70 MB weights
        repo_copy = Path(__file__).resolve().parents[1] / weights
        if not Path(weights).exists() and repo_copy.exists():
            weights = str(repo_copy)
        self.model = YOLO(weights)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.names = self.model.names
        self.tracks = {}
        self._person_tids = set()
        self._ignored = set()  # unknowns the operator dismissed ("don't track this")

    def warm(self):
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        with autorelease_pool():
            self.model.predict(dummy, device=self.device, imgsz=IMGSZ,
                               verbose=False)

    def ignore(self, tid):
        """Stop tracking one object for the rest of the session — an unknown
        the operator doesn't want to teach. Sticky per track id, like the
        person suppression, so it stays dismissed frame after frame."""
        self._ignored.add(tid)
        self.tracks.pop(tid, None)

    def reset(self):
        self.tracks.clear()
        self._person_tids.clear()
        self._ignored.clear()
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    def process(self, frame, now=None):
        """Run detection on one frame; returns live stable tracks."""
        now = now or time.time()
        with autorelease_pool():
            results = self.model.track(
                frame,
                device=self.device,
                conf=self.conf,
                imgsz=IMGSZ,
                max_det=MAX_DET,
                agnostic_nms=True,
                persist=True,
                verbose=False,
                # ultralytics >= 8.4 defaults to a tracker that attaches no ids
                tracker="botsort.yaml",
            )[0]
        h, w = frame.shape[:2]
        seen_tids = set()
        boxes = results.boxes
        polys = results.masks.xy if results.masks is not None else None
        if boxes is not None and boxes.id is not None:
            rows = zip(
                boxes.id.int().tolist(),
                boxes.cls.int().tolist(),
                boxes.conf.tolist(),
                boxes.xyxy.tolist(),
            )
            for i, (tid, cls, conf, box) in enumerate(rows):
                if tid in self._person_tids or tid in self._ignored:
                    continue
                # person check runs before the area band so an oversized face
                # box still poisons its track id for later, smaller frames
                if is_person_like(str(self.names.get(cls, ""))):
                    self._person_tids.add(tid)
                    self.tracks.pop(tid, None)
                    continue
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1) / (w * h)
                if not MIN_AREA <= area <= self.max_area:
                    continue
                t = self.tracks.setdefault(tid, Track(tid))
                t.frames += 1
                t.last_seen = now
                t.box = box
                # attention: size x centrality, so the object held to the
                # middle of the frame beats larger off-center clutter
                cx = (x1 + x2) / 2 / w - 0.5
                cy = (y1 + y2) / 2 / h - 0.5
                t.salience = (area ** 0.5) * (1 - (cx * cx + cy * cy) ** 0.5)
                # keep the sharpest crop since the last memory query
                q = crop_quality(frame, box, conf)
                if q >= t.crop_q or t.crop is None:
                    mask = (polys[i] if polys is not None and i < len(polys)
                            else None)
                    t.crop = padded_crop(frame, box, mask)
                    t.crop_q = q
                seen_tids.add(tid)

        if len(self._person_tids) > 4096:  # ids only grow; keep recent flags
            self._person_tids = set(sorted(self._person_tids)[-1024:])

        for tid, t in list(self.tracks.items()):
            if tid not in seen_tids:
                if now - t.last_seen > DEAD_SECONDS:
                    del self.tracks[tid]
                else:
                    t.frames = 0  # streak broken, must restabilize

        return [t for t in self.tracks.values() if t.stable]
