"""The robot's loop, minus the camera: capture → detect → embed → match → teach.

Both front ends drive this class — the live webcam app and the headless
file mode — so the smoke test exercises the same code the shoot records.
"""
import time
from datetime import date
from pathlib import Path

import cv2

from robot import audio, models
from robot.detect import Detector
from robot.memory import Memory, RECOGNIZE_THRESHOLD


def day_start_ts():
    return time.mktime(date.today().timetuple())


class Robot:
    def __init__(self, data_dir="edge-data", weights="yoloe-11l-seg-pf.pt",
                 threshold=RECOGNIZE_THRESHOLD, conf=None, max_area=None):
        self.memory = Memory(data_dir, threshold=threshold)
        opts = {k: v for k, v in
                (("conf", conf), ("max_area", max_area)) if v}
        self.detector = Detector(weights, **opts)
        self.thumbs = Path(data_dir) / "thumbs"
        self.thumbs.mkdir(exist_ok=True)
        self.events = []  # recent memory writes, for the on-screen log

    def log(self, msg):
        self.events.append(f"{time.strftime('%H:%M:%S')}  {msg}")
        del self.events[:-8]

    def _thumb(self, crop, tag):
        path = self.thumbs / f"{time.time_ns()}_{tag}.jpg"
        cv2.imwrite(str(path), crop)
        return str(path)

    # -- the loop --------------------------------------------------------------

    def process_frame(self, frame, now=None):
        """Detect, embed and match due tracks, pick what the robot attends to.

        Every recognized object stays in view, but only ONE unknown at a
        time — the most prominent — is shown, remembered, and teachable.
        Other unknowns are ignored: they're clutter, not candidates.
        """
        now = now or time.time()
        tracks = self.detector.process(frame, now)
        for t in tracks:
            if not t.due_for_query(now):
                continue
            t.vec = models.embed_crop(t.crop)
            hit, score = self.memory.recognize(t.vec)
            t.score = score
            t.label = hit.payload["label"] if hit else None
            t.note = hit.payload["transcript"] if hit else None
            t.last_query = now
            t.crop_q = 0.0  # collect a fresh best crop for the next query

        knowns = [t for t in tracks if t.label]
        primary = self.focused(
            [t for t in tracks if not t.label and t.last_query])
        display = knowns + ([primary] if primary else [])

        for t in display:
            if not t.sighted and t.vec is not None:
                self.memory.remember_sighting(
                    t.vec, t.label or "unknown", ts=now,
                    thumb=self._thumb(t.crop, t.tid),
                )
                t.sighted = True
                self.log(f"seen: {t.label or 'unknown'} ({t.score:.2f})")
        return display

    @staticmethod
    def focused(tracks):
        """The teach/recognize subject: the most salient stable thing in
        view — size weighted by centrality, so the object held to the
        middle of the frame wins over larger off-center clutter."""
        return max(tracks, key=lambda t: t.salience) if tracks else None

    # -- teach -----------------------------------------------------------------

    def teach(self, crop, wav_path):
        """One spoken sentence → one point carrying BOTH named vectors."""
        transcript = models.transcribe(wav_path)
        label = audio.parse_label(transcript)
        pid = self.memory.teach(
            image_vec=models.embed_crop(crop),
            text_vec=models.embed_text(transcript),
            label=label,
            transcript=transcript,
            thumb=self._thumb(crop, "taught"),
        )
        self.log(f'taught "{label[:18]}" -> image + text')
        return {"id": pid, "label": label, "transcript": transcript}

    # -- ask -------------------------------------------------------------------

    def ask(self, question, since_ts=None):
        """Cross-modal, time-filtered recall — grouped seen vs heard."""
        return self.memory.day_recall(
            text_vec=models.embed_query(question),
            clip_text_vec=models.embed_query_clip(question),
            since_ts=day_start_ts() if since_ts is None else since_ts,
        )

    def ask_from_wav(self, wav_path, since_ts=None):
        question = models.transcribe(wav_path)
        return question, self.ask(question, since_ts)

    # -- the reboot beat ---------------------------------------------------------

    def reboot(self):
        """Close the shard, reload from disk. Recognition state resets too."""
        self.memory.reopen()
        self.detector.reset()
        n = self.memory.count()
        self.log(f"shard reopened from disk — {n} memories")
        return n

    def close(self):
        self.memory.close()
