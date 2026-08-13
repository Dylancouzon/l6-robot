"""Detection and cadence — the two "new pieces" L6 names but doesn't teach.

YOLOE (prompt-free) finds and crops objects; its labels are discarded
entirely — detection finds *a thing*, memory tells it *which* thing. Nothing
here reads a class name any more: the detector's own vocabulary is not
consulted for anything, which is the cleanest statement of that idea the code
has ever made.

Cadence: a track must be stable (N consecutive frames) before it becomes a
match candidate or teachable, and it re-queries memory every couple of
seconds, not per frame. Design and all tuning constants lifted from
qdrant-labs/memory-fleet, where they were field-tuned on live desk scenes.
"""
import os
import time

import cv2
import numpy as np

from robot import config

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # no pip calls at runtime

try:
    # torch-MPS autoreleases Metal objects per inference; a pure-Python loop
    # never drains the pool, leaking ~80 MB/min. Every model call must run
    # inside this pool so the objects are released each iteration.
    from objc import autorelease_pool
except ImportError:  # non-macOS: nothing to drain
    from contextlib import nullcontext as autorelease_pool

CONF = config.DETECT_CONF   # per-camera, see .env (--conf overrides)
IMGSZ = 640
MAX_DET = 64
# Normalized box-area band: drops speck noise AND the oversized phantom regions
# a prompt-free detector emits for walls, desks and whole rooms. Demo objects
# are hand-held scale, so the cap stays tight.
#
# Since people stopped being filtered out, this cap is also what decides how
# close a person can stand and still be teachable. Measured on this camera: a
# seated person at desk distance is 0.072-0.078 of the frame, comfortably
# inside the cap, so no calibration change was needed. A face filling the frame
# is not — back off, or raise DETECT_MAX_AREA and accept more phantom boxes.
MIN_AREA = 0.0008
MAX_AREA = config.DETECT_MAX_AREA   # per-camera, see .env (--max-area)
STABLE_FRAMES = 3
REQUERY_SECONDS = 2.0
# How long a track survives with no detection before it is deleted outright.
# Generous on purpose, and it is NOT the box's lifetime — see the `frames` decay
# at the bottom of `process`, which hides the box after two missed passes
# regardless. This only decides whether the object comes back as ITSELF.
#
# Measured on the live board: a real object's detector confidence swings either
# side of DETECT_CONF (the microphone: 0.34-0.91, absent altogether on some
# frames), so dropouts past a second are routine. At 1.5 s the Track was
# destroyed, taking `label`, `note`, `thumb`, `vec` and `last_query` with it, and
# the object returned as a stranger: no name for STABLE_FRAMES passes, then a
# fresh embed and a fresh "seen" write. That is what the operator saw as a
# recognized object "coming and going" on a static scene — and the shard counted
# it, 57 Track births across 17 tracker ids in 25 minutes. BoT-SORT's own
# track_buffer is 30 frames (~15 s of passes here), so 1.5 s was far stricter
# than the tracker underneath it.
DEAD_SECONDS = 5.0
PAD = 0.12         # small margin; the mask removes the background anyway
FILL = (124, 124, 124)

# People used to be filtered out here, by matching the detector's class name
# against a word list. That is gone on purpose: a person is now just another
# thing to teach and recognize, so "this is Dylan" works exactly like "this is
# my laptop". See "People are objects now" in CLAUDE.md for what that costs
# (hands and faces become teachable clutter) and what pays it back.


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
        self.guess = None      # nearest taught label just under the bar —
                               # display only, never recognition
        self.note = None       # the taught transcript, recalled on match
        self.thumb = None      # the matched taught view, shown next to the live crop
        self.score = 0.0
        self.vec = None        # last CLIP embedding of the crop
        self.sighted = False   # one "seen" memory per track, not per frame

    @property
    def stable(self):
        return self.frames >= STABLE_FRAMES

    def due_for_query(self, now):
        return self.stable and now - self.last_query >= REQUERY_SECONDS

    def requery_now(self):
        """Make this track due at the next detect pass without pretending it
        was never queried — clearing `last_query` outright blanks the panel to
        "looking..." for a beat, which reads as a glitch right after a teach."""
        self.last_query = time.time() - REQUERY_SECONDS


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
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.tracks = {}
        # unknowns the operator dismissed ("don't track this"), tid -> a
        # small block record. This is only the per-frame gate: the durable
        # record of a dismissal is a kind="ignored" point in the shard (see
        # Memory.ignore), which is what the memory tab lists and what
        # re-suppresses the same-looking object after a restart, when these
        # ids mean nothing. Each block carries the pid of the point it came
        # from, so deleting that point can lift its blocks too (unblock).
        #
        # Copy-on-write: mutators REPLACE the dict rather than editing it in
        # place. All mutators run under the app's live-state lock (Q on the
        # key thread, /unignore on an HTTP thread, the requery re-suppression
        # on the detect thread), so writes never race each other — and a
        # reader that grabs the reference iterates a snapshot that can no
        # longer change.
        self._ignored = {}

    def warm(self):
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        with autorelease_pool():
            self.model.predict(dummy, device=self.device, imgsz=IMGSZ,
                               verbose=False)

    def ignore(self, tid, pid=None):
        """Stop tracking one object — an unknown the operator doesn't want to
        teach, or one that matched a persisted dismissal at requery. Sticky
        per track id, so it stays dismissed frame after frame however the
        classifier flickers. `pid` names the shard point behind the block."""
        ignored = dict(self._ignored)  # mutate the copy, then publish it
        ignored[tid] = {"tid": tid, "pid": pid}
        self._ignored = ignored
        self.tracks.pop(tid, None)

    def unblock(self, pid):
        """Lift every block that came from one deleted ignore point. The
        object comes back on its own — a blocked track carries no state to
        restore: if the tracker still holds an id it resumes at the next
        pass, and otherwise the object is re-proposed under a new one."""
        self._ignored = {k: v for k, v in self._ignored.items()
                         if v.get("pid") != pid}

    def reset(self):
        self.tracks.clear()
        self._ignored = {}  # replaced, never mutated — see __init__
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
                # capped one above the gate: enough for the streak below to
                # absorb a single blink, and no more. Every extra frame of
                # credit is a frame where a departed object still has a box
                # and a stale crop, and could be taught by mistake.
                t.frames = min(t.frames + 1, STABLE_FRAMES + 1)
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

        for tid, t in list(self.tracks.items()):
            if tid not in seen_tids:
                if now - t.last_seen > DEAD_SECONDS:
                    del self.tracks[tid]
                else:
                    # decay, don't reset: the detector blinks on single frames,
                    # and a reset hides an established box for STABLE_FRAMES
                    # more passes, which is most of the on-screen flicker.
                    # Two misses in a row still drop it (see the cap above).
                    # This, not DEAD_SECONDS, is the box's lifetime — which is
                    # why raising DEAD_SECONDS cannot widen the window in which
                    # a departed object still carries a teachable stale crop.
                    t.frames = max(0, t.frames - 1)

        return [t for t in self.tracks.values() if t.stable]
