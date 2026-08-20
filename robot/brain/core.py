"""The robot's loop, minus the camera: detect -> embed -> match -> teach -> recall.

`Robot` is the whole brain. It owns the detector and the memory, and both front
ends drive it - the live app and the headless replay - so nothing about a
webcam, a browser or a button appears in this file.
"""
import time
from datetime import date
from pathlib import Path

import cv2

from robot.brain import labels, models
from robot.brain.detect import Detector
from robot.brain.memory import Memory, RECOGNIZE_THRESHOLD


def day_start_ts():
    return time.mktime(date.today().timetuple())


# Attention hysteresis. Salience is recomputed every frame from box geometry,
# so two similar objects otherwise trade the box several times a second, and
# you cannot teach a target that holds still for 100 ms. A challenger has to be
# this much more salient than the incumbent to take focus from it.
#
# This is the one number to turn if the box feels twitchy. Raising it makes
# focus calmer and also harder to move on purpose.
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
        # What the robot is attending to. Decided once per detect pass, on the
        # thread that owns track state, so the front end reads a decision
        # instead of recomputing one on every rendered frame.
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
        """Save what the box was around, plus a margin: the memory tab's picture.

        Context, not evidence. The frame is the one on screen when the button
        went down, but the box comes from the last completed detect pass, so on
        a moving object this can trail what was actually learned.
        """
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
        # quality 80, unlike the embedded crops: a page of these is fetched
        # over the robot's own hotspot, and nothing scores against them
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
                # a near-miss wears an orange "hat?" box instead of UNKNOWN.
                # Display only: a guessed track is still unlabeled, so it is
                # still the teach target and still writes no sighting.
                t.guess = guess
                t.note = hit.payload["transcript"] if hit else None
                t.thumb = hit.payload.get("thumb") if hit else None
                t.last_query = now
                t.crop_q = 0.0  # collect a fresh best crop for the next query
                if hit is None:
                    # An unknown that looks like something the operator
                    # dismissed is dismissed again, silently. This is what
                    # makes IGNORE survive a restart. Checked only when no
                    # taught view matched, so a taught label always wins.
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
            # only recognized objects are logged: an unnamed thing has no label
            # to recall, so a sighting for it would just clutter recall
            if t.label and not t.sighted and t.vec is not None:
                self.memory.remember_sighting(
                    t.vec, t.label, ts=now,
                    thumb=self._scene(frame, t.box),
                )
                t.sighted = True
                self.log(f"seen: {t.label} ({t.score:.2f})")
        return display

    def _one_per_label(self, tracks):
        """At most one box per remembered object, keeping the best match.

        Teaching an object two or three times is the recommended habit, and
        every taught view is a memory the detector's other boxes can match too
        - teach yourself twice and the person box, face box and torso box all
        come back with your name. One memory, one box.

        Best is highest score, then salience, EXCEPT that the track holding
        attention keeps its box. That exception is load-bearing: dropping a
        track out of `display` takes it out of pick_focus()'s candidate list,
        and pick_focus() can only retain an incumbent that is still a candidate - so
        without it a score flip bypasses FOCUS_MARGIN entirely and attention
        re-derives by raw salience, which can land on a third object.

        The losers keep their labels and tracking. They are simply not drawn
        and not written as sightings, which also stops one object logging a
        sighting per box. Two identical mugs in frame therefore show one box
        between them: the labels are the same string, so the robot has nothing
        to tell them apart with.
        """
        # last pass's attention, since this runs before pick_focus() picks the
        # next one. None or an unknown simply never matches a labelled track.
        attending = self._incumbent.get("view")
        best = {}
        for t in tracks:
            key = t.label.lower()
            cur = best.get(key)
            if cur is None or t is attending:
                best[key] = t
            elif cur is not attending and (t.score, t.salience) > (cur.score,
                                                                  cur.salience):
                best[key] = t
        return list(best.values())

    def pick_focus(self, tracks, pool):
        """The most salient thing in view: size weighted by centrality, so an
        object held to the middle of the frame wins over larger off-center
        clutter.

        Sticky, by FOCUS_MARGIN. `pool` names the candidate set, so the whole
        view and the unknowns-only set each keep their own incumbent.

        The incumbent is held as the Track object, never its id: the tracker
        restarts ids from 1 on reset, so an id would let a dead object hand its
        focus to whatever new track inherits the number.
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

    def teach(self, crop, transcript, frame=None, box=None):
        """One spoken sentence -> one point carrying both named vectors.

        Takes the transcript, not the WAV: speech-to-text is slow, so the
        caller runs it before claiming the live-state lock. `frame` and `box`
        only decorate the memory with a picture for the tab to show.

        Nothing here rejects a poor crop. A blank or reflective one makes a
        memory that matches most of the room, and a correctly calibrated
        threshold is what keeps that harmless - so if it happens, calibrate
        with testdata/verify_scores.py rather than filtering crops.
        """
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
        # labelled in place - the point just written IS this crop, so the match
        # is certain - which turns the box teal on the tap
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
            # a guess wearing the deleted name would outlive it otherwise
            if t.guess and t.guess.lower() == label.lower():
                t.guess = None
                t.requery_now()
        self.log(f'forgot "{label[:18]}" -> {n} point(s)')
        return n

    def rename(self, label, to, text_vec=None):
        """Fix an object's name without touching what it looks like.

        Renaming onto a name that already exists merges the two objects, which
        is the cure for "Laptop" and "laptop" having become separate things.

        `text_vec` is the new name embedded, which the caller computes OFF the
        live-state lock (a first-of-session Nomic load is ~4 s of held GIL,
        long enough for DEAD_SECONDS to delete live tracks). Passing it is what
        lets recall find the object by its corrected name; see `Memory.rename`.
        Optional, so a caller with no encoders - a script, a test - can still
        fix a spelling.
        """
        to = labels.norm_label(to)
        n = self.memory.rename(label, to, text_vec=text_vec)
        for t in self.detector.tracks.values():
            # renamed in place rather than requeried: the IMAGE vector did not
            # change, so the box can just start saying the new name
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

        Re-teaching adds a view rather than replacing one, so a blurred teach
        would otherwise sit in memory forever matching things it shouldn't.
        Dropping an object's LAST taught view forgets the whole object, since
        leaving its sightings behind would describe something the robot can no
        longer recognize.

        The point has to actually belong to this object, and that is checked
        here rather than trusted: the id and label arrive together from a page
        that may be looking at a stale list.
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
        """Dismiss an unknown so the robot stops offering it.

        Persistent: the dismissal writes the crop's vector to the shard, so the
        same-looking object is re-suppressed after a restart, when track ids
        mean nothing. Takes the track rather than its id so a picture can go
        with it - the memory tab lists these, and an undo button over no
        picture is not something anyone can use with confidence.
        """
        # the teach target has always been embedded already; this fallback is
        # here so a caller with a never-queried track cannot write a
        # vectorless point
        vec = track.vec if track.vec is not None else models.embed_crop(track.crop)
        pid = self.memory.ignore(vec, thumb=self._scene(frame, track.box))
        self.detector.ignore(track.tid, pid=pid)
        # cleared here too, not just in the detector: attention is decided once
        # per detect pass, so otherwise a dismissed object keeps its box and
        # panel row for up to half a second of a press that visibly did nothing
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
        """Recall, in three steps: the name the operator gave the thing, then
        what was SAID about it, then where it was last seen.

        The question and the taught transcripts live in the same text space, so
        Nomic picks the object reliably. And once the object is chosen, "where
        did I leave it" is a question about time, not similarity - the newest
        sightings, not the nearest vectors.

        "What did you see today?" is an inventory instead: the distinct objects
        sighted since morning, no vectors involved.
        ponytail: one phrase check, not intent parsing.
        """
        if "what did you see" in question.lower():
            since = day_start_ts() if since_ts is None else since_ts
            return {"inventory": True, "label": None, "note": None,
                    "score": 0.0, "sightings": self.memory.seen_since(since)}
        # Two chances, in this order: the question NAMES some objects, or it
        # merely sounds like one. Names decide which objects are CANDIDATES -
        # the operator's own words for things, so nothing unnamed can win - and
        # the cosine then chooses among them, which is what stops "did dylan
        # take my smartphone?" answering with the person. It is also what makes
        # a shard whose transcripts drifted from their labels answer correctly,
        # with nothing to migrate. Name nothing and the vector picks outright,
        # exactly as before (a misheard question carries no label at all).
        qv = models.embed_query(question)
        named = self.memory.names_in(question)
        hit = self.memory.best_taught(qv, labels=named)
        if hit is None and named:
            # forgotten between the scan and the query; answer as if unnamed
            hit = self.memory.best_taught(qv)
        if hit is None:
            return {"inventory": False, "label": None, "note": None,
                    "score": 0.0, "sightings": []}
        label = hit.payload.get("label")
        return {"inventory": False, "label": label,
                "note": hit.payload.get("transcript"),
                "score": hit.score,
                # a teach counts as an occasion here: it carries a ts, a place
                # and a picture, so an object taught today and not recognized
                # since answers with today, not with last week. Sightings-only
                # is still what the memory tab reads.
                "sightings": self.memory.last_sightings(
                    label, kinds=("seen", "taught"))}

    # -- the reboot beat ---------------------------------------------------------

    def reboot(self):
        """Close the shard, reload it from disk. Recognition state resets too."""
        t0 = time.perf_counter()
        self.memory.reopen()
        ms = (time.perf_counter() - t0) * 1000
        self.detector.reset()
        # tracking state is gone, so the panel must not keep showing a subject
        # from before the reboot
        self.attention = self.teachable = None
        n = self.memory.count()
        self.log(f"shard reopened in {ms:.0f} ms — {n} memories")
        return n, ms

    def close(self):
        self.memory.close()
