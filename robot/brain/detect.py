"""Find and track objects before looking them up in memory.

YOLOE supplies boxes and masks. The robot ignores its class labels because
memory, not the detector, names objects.
"""
import os
import time

import cv2
import numpy as np

from robot import config

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # no pip calls at runtime

try:
    # torch-MPS autoreleases Metal objects per inference, and a plain Python
    # loop never drains the pool. macOS only; nothing to drain elsewhere.
    from objc import autorelease_pool
except ImportError:
    from contextlib import nullcontext as autorelease_pool

CONF = config.DETECT_CONF           # per-camera, see .env (--conf overrides)
MIN_AREA = config.DETECT_MIN_AREA   # per-camera, see .env
MAX_AREA = config.DETECT_MAX_AREA   # per-camera, see .env (--max-area)
IMGSZ = 640
MAX_DET = 64

STABLE_FRAMES = 3      # passes before a track is displayed or teachable
REQUERY_SECONDS = 2.0  # how often a stable track re-asks memory

# Keep track state across short detector dropouts. The `frames` counter below
# hides a missed box sooner than this state expires.
DEAD_SECONDS = 5.0

PAD = 0.12          # crop margin; the mask removes the background anyway
FILL = (124, 124, 124)


def padded_crop(frame, box, mask=None):
    """The crop that gets embedded: 12% margin, background flattened to gray.

    Filling everything outside the segmentation mask is what makes recognition
    survive a changed background or a hand holding the object.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    region = frame[y1:y2, x1:x2]
    if mask is not None and len(mask) >= 3 and region.size:
        full = np.zeros((h, w), np.uint8)
        cv2.fillPoly(full, [np.asarray(mask, dtype=np.int32)], 255)
        # grow the mask slightly first: the segmentation edge cuts just inside
        # the object, and filling right up to it shaves off its outline
        full = cv2.dilate(full, np.ones((7, 7), np.uint8), iterations=1)
        inside = full[y1:y2, x1:x2] > 0
        region = region.copy()
        region[~inside] = FILL
    return region


def crop_quality(frame, box, conf):
    """How good a portrait of an object this crop is: bigger, sharper, surer."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    small = cv2.resize(frame[y1:y2, x1:x2], (96, 96),
                       interpolation=cv2.INTER_AREA)
    sharp = cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                          cv2.CV_32F).var()
    area = (x2 - x1) * (y2 - y1) / (w * h)
    # sharpness is capped so one noisy high-contrast crop cannot outrank a
    # bigger, better-framed one on that term alone
    return conf * (area ** 0.5) * min(sharp, 1200.0)


class Track:
    """One object the detector is following, and what memory said about it."""

    def __init__(self, tid):
        self.tid = tid
        # A stability score that rises on detection and falls on a miss.
        self.frames = 0
        self.last_seen = 0.0
        self.last_query = 0.0
        self.box = None
        self.crop = None       # best (sharpest) padded crop since last query
        self.crop_q = 0.0
        self.salience = 0.0    # size x centrality: what the robot attends to
        self.label = None      # from memory, never from the detector
        self.guess = None      # near match for display only
        self.note = None       # the taught transcript, recalled on match
        self.thumb = None      # the matched taught view, shown beside the crop
        self.score = 0.0
        self.vec = None        # last CLIP embedding of the crop
        self.sighted = False   # one "seen" memory per track, not per frame

    @property
    def stable(self):
        return self.frames >= STABLE_FRAMES

    def due_for_query(self, now):
        return self.stable and now - self.last_query >= REQUERY_SECONDS

    def requery_now(self):
        """Make this track due at the next detect pass. Not `last_query = 0`,
        which is falsy and blanks the panel to "looking..." for a beat."""
        self.last_query = time.time() - REQUERY_SECONDS


class Detector:
    """YOLOE + BoT-SORT tracking + the stability gate."""

    def __init__(self, weights="yoloe-11l-seg-pf.pt", conf=CONF,
                 max_area=MAX_AREA):
        from pathlib import Path

        import torch
        from ultralytics import YOLO
        self.conf = conf
        self.max_area = max_area
        # Reuse weights stored at the repository root from any working directory.
        repo_copy = Path(__file__).resolve().parents[2] / weights
        if not Path(weights).exists() and repo_copy.exists():
            weights = str(repo_copy)
        self.model = YOLO(weights)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.tracks = {}
        # Live track blocks. Persisted ignore vectors survive track-id changes.
        # Replacing this dict gives lock-free readers a consistent snapshot.
        self._ignored = {}

    def warm(self):
        """Run one dummy inference so the first real frame is not the slow one."""
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        with autorelease_pool():
            self.model.predict(dummy, device=self.device, imgsz=IMGSZ,
                               verbose=False)

    def ignore(self, tid, pid=None):
        """Stop tracking one object. Sticky per track id, so it stays dismissed
        however the detector flickers. `pid` names the shard point behind it."""
        ignored = dict(self._ignored)
        ignored[tid] = {"tid": tid, "pid": pid}
        self._ignored = ignored
        self.tracks.pop(tid, None)

    def unblock(self, pid):
        """Lift every block that came from one deleted ignore point. The object
        comes back on its own: a blocked track carries no state to restore."""
        self._ignored = {k: v for k, v in self._ignored.items()
                         if v.get("pid") != pid}

    def reset(self):
        self.tracks.clear()
        self._ignored = {}  # replaced, never mutated - see __init__
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    def process(self, frame, now=None):
        """Run detection on one frame; returns the live stable tracks."""
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
            # no class column: the detector's labels are not read at all
            rows = zip(
                boxes.id.int().tolist(),
                boxes.conf.tolist(),
                boxes.xyxy.tolist(),
            )
            for i, (tid, conf, box) in enumerate(rows):
                if tid in self._ignored:
                    continue
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1) / (w * h)
                if not MIN_AREA <= area <= self.max_area:
                    continue
                t = self.tracks.setdefault(tid, Track(tid))
                # One extra point of credit absorbs a single missed detection.
                t.frames = min(t.frames + 1, STABLE_FRAMES + 1)
                t.last_seen = now
                t.box = box
                # attention: size x centrality, so an object held to the middle
                # of the frame beats larger off-center clutter
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

        for tid, t in list(self.tracks.items()):
            if tid not in seen_tids:
                if now - t.last_seen > DEAD_SECONDS:
                    del self.tracks[tid]
                else:
                    # Decay gradually so one missed detection does not flicker.
                    t.frames = max(0, t.frames - 1)

        return [t for t in self.tracks.values() if t.stable]
