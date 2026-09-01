"""The robot's detection, recognition, teaching, and recall logic."""
import time
from datetime import date
from pathlib import Path

import cv2

from robot.brain import labels, models
from robot.brain.detect import Detector
from robot.brain.memory import RECOGNIZE_THRESHOLD, Memory


def day_start_ts():
    # Recall uses the operator's local calendar day.
    return time.mktime(date.today().timetuple())  # noqa: DTZ011


# A new object must be this much more prominent to take focus. Raising the
# value makes focus steadier but harder to move.
FOCUS_MARGIN = 1.6

# The picture kept beside every memory for a human to look at: the detector's
# box plus a wider margin than the recognition crop, unmasked. Nothing is
# scored against it.
SCENE_PAD = 0.3
SCENE_PX = 480   # longest side; a phone card shows it about 170 px wide


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
        # Focus is decided once per detection pass and shared with the UI.
        self.attention = None  # panel subject, and the FORGET target
        self.teachable = None  # the salient UNKNOWN, and the TEACH target
        self._incumbent = {}   # pool name -> the Track currently holding focus

    def log(self, msg):
        self.events.append(f"{time.strftime('%H:%M:%S')}  {msg}")
        del self.events[:-8]

    def _thumb(self, crop, tag, quality=None):
        path = self.thumbs / f"{time.time_ns()}_{tag}.jpg"
        opts = [] if quality is None else [cv2.IMWRITE_JPEG_QUALITY, quality]
        cv2.imwrite(str(path), crop, opts)
        return str(path)

    def _scene(self, frame, box):
        """Save the object and some context for the memory tab."""
        if frame is None or not frame.size:
            return None
        h, w = frame.shape[:2]
        img = frame
        if box is not None:
            x1, y1, x2, y2 = box
            px, py = (x2 - x1) * SCENE_PAD, (y2 - y1) * SCENE_PAD
            x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
            x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
            region = frame[y1:y2, x1:x2]
            if region.size:
                img = region
        h, w = img.shape[:2]
        scale = SCENE_PX / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (max(1, int(w * scale)),
                                   max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        # These images are for display, so smaller JPEGs are sufficient.
        return self._thumb(img, "scene", quality=80)

    # -- the loop --------------------------------------------------------------

    def process_frame(self, frame, now=None):
        """Detect, embed and match due tracks, then pick what to attend to.

        Every recognized object stays in view, but only one box per remembered
        object, and only one unknown - the most prominent - is shown and
        teachable. Other unknowns are clutter, not candidates.
        """
        now = now or time.time()
        tracks = self.detector.process(frame, now)
        live = []
        for t in tracks:
            if t.due_for_query(now):
                t.vec = models.embed_crop(t.crop)
                hit, score, guess = self.memory.recognize(t.vec)
                t.score = score
                t.label = hit.payload["label"] if hit else None
                # Near matches are displayed as guesses, never as recognition.
                t.guess = guess
                t.note = hit.payload["transcript"] if hit else None
                t.thumb = hit.payload.get("thumb") if hit else None
                t.last_query = now
                t.crop_q = 0.0  # collect a fresh best crop for the next query
                if hit is None:
                    # Taught memories take priority over persisted ignores.
                    ig = self.memory.match_ignored(t.vec)
                    if ig is not None:
                        self.detector.ignore(t.tid, pid=ig.id)
                        continue  # not displayed, not teachable
            live.append(t)

        knowns = self._one_per_label([t for t in live if t.label])
        primary = self.pick_focus(
            [t for t in live if not t.label and t.last_query], pool="unknown")
        display = knowns + ([primary] if primary else [])
        # The teach target is the salient unknown, never a known: otherwise a
        # recognized object could steal focus and get relabeled.
        self.teachable = primary
        self.attention = self.pick_focus(display, pool="view")

        for t in display:
            # Only named objects produce recallable sightings.
            if t.label and not t.sighted and t.vec is not None:
                self.memory.remember_sighting(
                    t.vec, t.label, ts=now,
                    thumb=self._scene(frame, t.box),
                )
                t.sighted = True
                self.log(f"seen: {t.label} ({t.score:.2f})")
        return display

    def _one_per_label(self, tracks):
        """Keep one box per label, preferring the current focus or best match."""
        # Preserve the current focus so small score changes do not move its box.
        attending = self._incumbent.get("view")
        best = {}
        for t in tracks:
            key = t.label.lower()
            cur = best.get(key)
            if cur is None or t is attending:  # noqa: SIM114
                best[key] = t
            elif cur is not attending and (t.score, t.salience) > (cur.score,
                                                                  cur.salience):
                best[key] = t
        return list(best.values())

    def pick_focus(self, tracks, pool):
        """Choose a prominent object while keeping focus reasonably stable."""
        if not tracks:
            return None
        best = max(tracks, key=lambda t: t.salience)
        held = self._incumbent.get(pool)
        if held in tracks and best.salience < held.salience * FOCUS_MARGIN:
            best = held
        self._incumbent[pool] = best
        return best

    # -- teach -----------------------------------------------------------------

    def teach(self, crop, transcript, frame=None, box=None):
        """Store one spoken description with image and text vectors."""
        label = labels.parse_label(transcript)
        pid = self.memory.teach(
            image_vec=models.embed_crop(crop),
            text_vec=models.embed_text(transcript),
            label=label,
            transcript=transcript,
            thumb=self._thumb(crop, "taught"),
            scene=self._scene(frame, box),
        )
        self.log(f'taught "{label[:18]}" -> image + text')
        return {"id": pid, "label": label, "transcript": transcript}

    def confirm(self, label, frame=None):
        """One tap on the orange guess: teach the attending crop as that name.

        Everything a teach needs is already on screen - the crop and the name -
        so this skips the voice round trip. `label` is what the page displayed
        at tap time and is re-checked against the live guess before acting,
        because focus moves under a finger already on its way down. Returns
        None if the guess moved on, having written nothing.
        """
        t = self.attention
        if (t is None or t.label or not t.guess or t.crop is None
                or t.guess.lower() != label.lower()):
            return None
        res = self.teach(t.crop, t.guess, frame=frame, box=t.box)
        # Update the box immediately because this exact crop was just taught.
        t.label = res["label"]
        t.note = res["transcript"]
        t.guess = None
        t.requery_now()
        return res

    # -- forget / ignore -------------------------------------------------------

    def forget(self, label):
        """Delete what the robot knows about one object.

        Also strips the name off anything still wearing it in view, so the box
        turns red on the press instead of at the next requery.
        """
        n = self.memory.forget(label)
        for t in self.detector.tracks.values():
            if t.label and t.label.lower() == label.lower():
                t.label = t.note = t.thumb = None
                t.requery_now()
            # Clear guesses that refer to the deleted object.
            if t.guess and t.guess.lower() == label.lower():
                t.guess = None
                t.requery_now()
        self.log(f'forgot "{label[:18]}" -> {n} point(s)')
        return n

    def rename(self, label, to, text_vec=None):
        """Fix an object's name without touching what it looks like.

        Renaming onto a name that already exists merges the two objects, which
        also merges names that differ only by capitalization. The optional
        `text_vec` lets recall search for the corrected name.
        """
        to = labels.norm_label(to)
        n = self.memory.rename(label, to, text_vec=text_vec)
        for t in self.detector.tracks.values():
            # The image vector is unchanged, so the visible label can update now.
            if t.label and t.label.lower() == label.lower():
                t.label = to
        self.log(f'renamed "{label[:14]}" -> "{to[:14]}" ({n} point(s))')
        return n

    def set_where(self, place):
        """Move the robot: change the place stamped on memories from now on."""
        where = self.memory.set_where(place)
        self.log(f'here: "{where}"' if where else "location cleared")
        return where

    def forget_view(self, pid, label):
        """Delete one taught view, or one sighting occasion. Returns (n, whole).

        Dropping the last taught view forgets the whole object. The point and
        label are checked together because the browser may show stale data.
        """
        ids = self.memory.taught_ids(label)
        if pid in ids:
            if len(ids) <= 1:
                return self.forget(label), True
            self.memory.forget_point(pid)
            self.log(f'dropped one view of "{label[:18]}"')
            return 1, False
        if pid in self.memory.seen_ids(label):
            n = self.memory.forget_sighting_burst(pid)
            self.log(f'dropped {n} sighting photo(s) of "{label[:18]}"')
            return n, False
        return 0, False

    def ignore(self, track, frame=None):
        """Persist a dismissed unknown and stop showing its live track."""
        # Callers normally supply an embedded track, but keep this method safe.
        vec = track.vec if track.vec is not None else models.embed_crop(track.crop)
        pid = self.memory.ignore(vec, thumb=self._scene(frame, track.box))
        self.detector.ignore(track.tid, pid=pid)
        # Clear the shared UI state instead of waiting for another detect pass.
        if self.attention is track:
            self.attention = None
        if self.teachable is track:
            self.teachable = None
        self.log("ignored one unknown")

    def unignore(self, pid):
        """Take one dismissal back: delete its point, lift its live blocks."""
        ok = self.memory.unignore(pid)
        if ok:
            self.detector.unblock(pid)
        return ok

    # -- ask -------------------------------------------------------------------

    def ask(self, question, since_ts=None):
        """Choose an object from the question and return its latest sightings.

        "What did you see" questions return today's distinct objects instead.
        """
        if "what did you see" in question.lower():
            since = day_start_ts() if since_ts is None else since_ts
            return {"inventory": True, "label": None, "note": None,
                    "score": 0.0, "sightings": self.memory.seen_since(since)}
        # Explicit object names narrow the semantic search when present.
        qv = models.embed_query(question)
        named = self.memory.names_in(question)
        hit = self.memory.best_taught(qv, labels=named)
        if hit is None and named:
            # The named object may have been forgotten between the two reads.
            hit = self.memory.best_taught(qv)
        if hit is None:
            return {"inventory": False, "label": None, "note": None,
                    "score": 0.0, "sightings": []}
        label = hit.payload.get("label")
        return {"inventory": False, "label": label,
                "note": hit.payload.get("transcript"),
                "score": hit.score,
                # Teaching is also a known time and place for recall.
                "sightings": self.memory.last_sightings(
                    label, kinds=("seen", "taught"))}

    # -- reload ----------------------------------------------------------------

    def reboot(self):
        """Close the shard, reload it from disk. Recognition state resets too."""
        t0 = time.perf_counter()
        self.memory.reopen()
        ms = (time.perf_counter() - t0) * 1000
        self.detector.reset()
        # Track objects are invalid after the detector resets.
        self.attention = self.teachable = None
        n = self.memory.count()
        self.log(f"shard reopened in {ms:.0f} ms: {n} memories")
        return n, ms

    def close(self):
        self.memory.close()
