"""Detection and cadence — the two "new pieces" L6 names but doesn't teach.

YOLOE (prompt-free) finds and crops objects; its labels are discarded —
detection finds *a thing*, memory tells it *which* thing. Class names are
consulted only to suppress people/hands/faces, and that suppression is
sticky per track because classifiers flicker.

Cadence: a track must be stable (N consecutive frames) before it becomes a
match candidate or teachable, and it re-queries memory every couple of
seconds, not per frame. Design lifted from qdrant-labs/memory-fleet.
"""
import time

CONF = 0.30
IMGSZ = 640
MAX_DET = 16
MIN_AREA = 0.002   # of frame, drops speckle
MAX_AREA = 0.55    # drops "the whole desk is one object"
STABLE_FRAMES = 3
REQUERY_SECONDS = 2.0
DEAD_SECONDS = 1.5
PAD = 0.12

PERSON_WORDS = ("person", "man", "woman", "human", "face", "hand", "arm",
                "finger", "head", "hair", "ear", "boy", "girl")


def padded_crop(frame, box):
    """Crop a detection box with 12% padding, clamped to the frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    return frame[y1:y2, x1:x2]


class Track:
    def __init__(self, tid):
        self.tid = tid
        self.frames = 0
        self.last_seen = 0.0
        self.last_query = 0.0
        self.box = None
        self.crop = None       # freshest padded crop (BGR)
        self.label = None      # from memory, never from the detector
        self.note = None       # the taught transcript, recalled on match
        self.score = 0.0
        self.sighted = False   # one "seen" memory per track, not per frame

    @property
    def stable(self):
        return self.frames >= STABLE_FRAMES

    def due_for_query(self, now):
        return self.stable and now - self.last_query >= REQUERY_SECONDS


class Detector:
    """YOLOE + BoT-SORT tracking + the stability gate."""

    def __init__(self, weights="yoloe-11l-seg-pf.pt"):
        from ultralytics import YOLO
        import torch
        self.model = YOLO(weights)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.names = self.model.names
        self.tracks = {}
        self._person_tids = set()

    def reset(self):
        self.tracks.clear()
        self._person_tids.clear()
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    def process(self, frame, now=None):
        """Run detection on one frame; returns live stable tracks."""
        now = now or time.time()
        results = self.model.track(
            frame,
            device=self.device,
            conf=CONF,
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
        if boxes is not None and boxes.id is not None:
            for tid, cls, box in zip(
                boxes.id.int().tolist(),
                boxes.cls.int().tolist(),
                boxes.xyxy.tolist(),
            ):
                name = str(self.names.get(cls, "")).lower()
                if any(word in name for word in PERSON_WORDS):
                    self._person_tids.add(tid)
                if tid in self._person_tids:
                    continue
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1) / (w * h)
                if not MIN_AREA <= area <= MAX_AREA:
                    continue
                t = self.tracks.setdefault(tid, Track(tid))
                t.frames += 1
                t.last_seen = now
                t.box = box
                t.crop = padded_crop(frame, box)
                seen_tids.add(tid)

        for tid, t in list(self.tracks.items()):
            if tid not in seen_tids:
                if now - t.last_seen > DEAD_SECONDS:
                    del self.tracks[tid]
                else:
                    t.frames = 0  # streak broken, must restabilize

        return [t for t in self.tracks.values() if t.stable]
