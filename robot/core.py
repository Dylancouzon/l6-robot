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


# Attention hysteresis. Salience is recomputed every frame from box geometry,
# so two objects of similar size and centrality otherwise trade the box back
# and forth several times a second — and you cannot reliably teach or forget a
# target that only holds still for 100 ms. A challenger must be this much more
# salient than the incumbent to take focus from it.
#
# Raised from 1.25 on an operator report that the box still traded between
# unknowns in a cluttered scene. 1.25 was picked by eye and tolerated salience
# swings up to 25%; on this camera a static object's box jitters more than that,
# because salience is size x centrality and the detector's box for one object
# changes size frame to frame. Baseline before the change, measured off
# /crop.jpg: 3.5 focus switches per minute, median dwell 2.34 s.
#
# A dwell timer was considered again and NOT built. It stays rejected for the
# reason recorded before — a second mechanism on one decision — plus one found
# by reading the code: any hold **gated on candidacy** is powerless against a
# blink, because `focused` can only retain an incumbent that is still in the
# candidate list and that list holds stable tracks only. A hold that returned a
# non-candidate would fix that half, and is a worse idea: it would point the
# panel, and TEACH, at an object that currently has no box and a stale crop.
#
# So the blink half of the churn is NOT fixed here, and raising the margin
# arguably makes it worse: a returning object must now beat whoever took its
# place by 60%. What softens it in practice is an accident worth knowing about —
# `focused` returns None on an empty candidate list *without* clearing the
# incumbent, so an object that blinks while nothing else is teachable simply
# resumes focus when it comes back.
FOCUS_MARGIN = 1.6


class Robot:
    def __init__(self, data_dir="edge-data", weights="yoloe-11l-seg-pf.pt",
                 threshold=RECOGNIZE_THRESHOLD, conf=None, max_area=None,
                 where=None):
        self.memory = Memory(data_dir, threshold=threshold, where=where)
        opts = {k: v for k, v in
                (("conf", conf), ("max_area", max_area)) if v is not None}
        self.detector = Detector(weights, **opts)
        self.thumbs = Path(data_dir) / "thumbs"
        self.thumbs.mkdir(exist_ok=True)
        self.events = []  # recent memory writes, for the on-screen log
        # What the robot is attending to, decided once per detect pass (see
        # process_frame) so the front end reads a decision instead of
        # recomputing one on every rendered frame.
        self.attention = None  # panel subject, and the FORGET target
        self.teachable = None  # the salient UNKNOWN, and the TEACH target
        self._incumbent = {}   # pool name -> the Track currently holding focus

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

        Every recognized object stays in view and is logged as a sighting;
        only ONE unknown at a time — the most prominent — is shown and
        teachable. Other unknowns are ignored: they're clutter, not candidates.
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
            t.thumb = hit.payload.get("thumb") if hit else None
            t.last_query = now
            t.crop_q = 0.0  # collect a fresh best crop for the next query

        knowns = [t for t in tracks if t.label]
        primary = self.focused(
            [t for t in tracks if not t.label and t.last_query], pool="unknown")
        display = knowns + ([primary] if primary else [])
        # Decided here, on the one thread that owns track state. The teach
        # target is the salient unknown, never a known — otherwise a
        # recognized object could steal focus and get relabeled.
        self.teachable = primary
        self.attention = self.focused(display, pool="view")

        for t in display:
            # only recognized objects are logged: an unnamed thing has no label
            # or text to recall, so a sighting for it would just clutter recall
            if t.label and not t.sighted and t.vec is not None:
                self.memory.remember_sighting(
                    t.vec, t.label, ts=now,
                    thumb=self._thumb(t.crop, t.tid),
                )
                t.sighted = True
                self.log(f"seen: {t.label} ({t.score:.2f})")
        return display

    def focused(self, tracks, pool):
        """The teach/recognize subject: the most salient stable thing in
        view — size weighted by centrality, so the object held to the
        middle of the frame wins over larger off-center clutter.

        Sticky, by FOCUS_MARGIN: whatever holds focus keeps it until a
        challenger is clearly more salient or its own track is gone. `pool`
        names the candidate set, so the whole view and the unknowns-only set
        each remember their own incumbent instead of overwriting each other.

        The incumbent is held as the Track object, not its id: the tracker
        restarts ids from 1 on reset, so an id would let a dead object hand
        its focus to whatever new track inherits its number.
        """
        if not tracks:
            return None
        best = max(tracks, key=lambda t: t.salience)
        held = self._incumbent.get(pool)
        if held in tracks and best.salience < held.salience * FOCUS_MARGIN:
            best = held
        self._incumbent[pool] = best
        return best

    # -- teach -----------------------------------------------------------------

    def teach(self, crop, transcript):
        """One spoken sentence → one point carrying BOTH named vectors.

        Takes the transcript, not the WAV: speech-to-text is slow, so the
        caller runs it before claiming the live-state lock.

        A crop with little information in it — a blank panel, a reflective
        surface, a blurred fragment — makes a memory that matches most of the
        room. Nothing here rejects one: a correctly calibrated threshold is
        what keeps it harmless, so if it happens, calibrate (see
        testdata/verify_scores.py) rather than filtering crops.
        """
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

    # -- forget / ignore -------------------------------------------------------

    def forget(self, label):
        """Delete what the robot knows about one recognized object (L2 forget).

        Also strips the label off anything still wearing it in view and makes
        those tracks re-ask memory at the next pass, so the box turns red on
        the press instead of at the next cadence tick — including tracks that
        are momentarily off-screen and would otherwise keep a deleted name.
        """
        n = self.memory.forget(label)
        for t in self.detector.tracks.values():
            if t.label == label:
                t.label = t.note = t.thumb = None
                t.requery_now()
        self.log(f'forgot "{label[:18]}" -> {n} point(s)')
        return n

    def ignore(self, tid):
        """Dismiss an unknown so the robot stops offering it — clutter you
        don't want to teach. Sticky per track, like person suppression."""
        self.detector.ignore(tid)

    # -- ask -------------------------------------------------------------------

    def ask(self, question, since_ts=None):
        """Cross-modal, time-filtered recall — grouped seen vs heard."""
        return self.memory.day_recall(
            text_vec=models.embed_query(question),
            clip_text_vec=models.embed_query_clip(question),
            since_ts=day_start_ts() if since_ts is None else since_ts,
        )

    # -- the reboot beat ---------------------------------------------------------

    def reboot(self):
        """Close the shard, reload from disk. Recognition state resets too."""
        t0 = time.perf_counter()
        self.memory.reopen()
        ms = (time.perf_counter() - t0) * 1000
        self.detector.reset()
        # tracking state is gone, so the panel must not keep showing a subject
        # from before the reboot; the next detect pass picks a fresh one
        self.attention = self.teachable = None
        n = self.memory.count()
        self.log(f"shard reopened in {ms:.0f} ms — {n} memories")
        return n, ms

    def close(self):
        self.memory.close()
