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
    def __init__(self, data_dir="edge-data", weights="yoloe-11l-seg-pf.pt"):
        self.memory = Memory(data_dir)
        self.detector = Detector(weights)
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
        """Detect, and for due tracks: embed the crop and match against memory."""
        now = now or time.time()
        tracks = self.detector.process(frame, now)
        for t in tracks:
            if not t.due_for_query(now):
                continue
            vec = models.embed_crop(t.crop)
            hit, score = self.memory.recognize(vec)
            t.score = score
            t.label = hit.payload["label"] if hit else None
            t.note = hit.payload["transcript"] if hit else None
            t.last_query = now
            if not t.sighted:
                self.memory.remember_sighting(
                    vec, t.label or "unknown", ts=now,
                    thumb=self._thumb(t.crop, t.tid),
                )
                t.sighted = True
                self.log(f"seen: {t.label or 'unknown'} ({score:.2f})")
        return tracks

    @staticmethod
    def focused(tracks):
        """The teach/recognize subject: the largest stable thing in view."""
        def area(t):
            x1, y1, x2, y2 = t.box
            return (x2 - x1) * (y2 - y1)
        return max(tracks, key=area) if tracks else None

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
        self.log(f'taught: "{label}" (vectors: image + text)')
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
