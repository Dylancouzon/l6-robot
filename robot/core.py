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

# The "scene" picture kept beside every taught crop: the object as the camera
# saw it, with a little room around it. The recognition crop is a poor way for
# a human to identify their own object in a list — it is masked, gray-filled
# and often a fragment — but so is a whole frame, where the object is a detail
# somewhere in a room. This is the middle: the detector's box plus a margin, so
# the object fills the picture and the margin says where it was.
#
# The margin is wider than the recognition crop's PAD (0.12) on purpose. That
# one is tuned for what CLIP should see; this one is tuned for what a person
# needs to recognize a thing at thumbnail size, and a little real background
# helps. Nothing scores against this picture.
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
        # What the robot is attending to, decided once per detect pass (see
        # process_frame) so the front end reads a decision instead of
        # recomputing one on every rendered frame.
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
        """Save what the box was around, plus a margin — the memory tab's picture.

        Unmasked and uncropped by the segmentation polygon, unlike the crop
        that gets embedded: this one exists to be looked at, so the real
        background inside the margin is the point rather than noise.

        The frame is the one on screen when the button went down, but the box
        is from the last completed detect pass — up to about half a second
        older, since a pass measures 0.1-0.6 s. On a moving object the crop can
        therefore trail it, and the embedded crop beside it is older still
        (a track keeps the sharpest crop since its last query). So this is not
        a picture of exactly what was learned. Context, not evidence.

        No box drawn on it: the picture IS the box now. Falls back to the whole
        frame when there is no box to crop to.
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
        # over the robot's own hotspot, and nothing scores against them.
        return self._thumb(img, "scene", quality=80)

    # -- the loop --------------------------------------------------------------

    def process_frame(self, frame, now=None):
        """Detect, embed and match due tracks, pick what the robot attends to.

        Every recognized object stays in view and is logged as a sighting —
        but only ONE box per remembered object (see `_one_per_label`). Only
        ONE unknown at a time — the most prominent — is shown and teachable.
        Other unknowns are ignored: they're clutter, not candidates.
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
                # a near-miss wears an orange "hat?" box instead of UNKNOWN —
                # display only: a guessed track is still unlabeled, so it is
                # still the teach target, writes no sighting, can't be
                # forgotten, and is still checked against the ignore list
                t.guess = guess
                t.note = hit.payload["transcript"] if hit else None
                t.thumb = hit.payload.get("thumb") if hit else None
                t.last_query = now
                t.crop_q = 0.0  # collect a fresh best crop for the next query
                if hit is None:
                    # An unknown that looks like something the operator
                    # dismissed is dismissed again, silently — this is what
                    # makes IGNORE survive a restart (and a REBOOT), since the
                    # tid blocks in the detector die with the run. Checked
                    # only when no taught view matched: a taught label always
                    # beats an ignore, so a degenerate ignored crop can hide
                    # unknown clutter at worst, never a taught object. Undo is
                    # TRACK AGAIN in the memory tab, same as ever.
                    ig = self.memory.match_ignored(t.vec)
                    if ig is not None:
                        self.detector.ignore(t.tid, pid=ig.id)
                        continue  # not displayed, not teachable
            live.append(t)

        knowns = self._one_per_label([t for t in live if t.label])
        primary = self.focused(
            [t for t in live if not t.label and t.last_query], pool="unknown")
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
                # the picture is the scene-style one (box + margin, unmasked),
                # same as the memory tab's: recall shows these rows to a
                # human, and the masked gray crop it used to store answers
                # "where did I leave it" with a fragment on a gray void
                self.memory.remember_sighting(
                    t.vec, t.label, ts=now,
                    thumb=self._scene(frame, t.box),
                )
                t.sighted = True
                self.log(f"seen: {t.label} ({t.score:.2f})")
        return display

    def _one_per_label(self, tracks):
        """At most one box per remembered object, keeping the best match.

        Teaching an object two or three times is the recommended habit — it is
        the single biggest recognition win — but every taught view is a memory
        the detector's other boxes can match too. Teach yourself twice and the
        person box, the face box and the torso box all come back "dylan", so
        the same memory is drawn three times over one human. One memory, one
        box.

        Best = highest recognition score, then salience — **except that the
        track currently holding attention keeps its box.** That exception is
        load-bearing, and it is not tidiness: dropping a track out of `display`
        takes it out of `focused`'s candidate list, and `focused` can only
        retain an incumbent that is still a candidate. So without it, a score
        flip between two boxes of one object silently bypasses FOCUS_MARGIN
        entirely — and attention does not move to the new winner, it
        re-derives by raw salience over what is left, which can be a *third*
        object. Observed with stubs: the panel jumped from "dylan" to a mug and
        back on the requery cadence, with the margin powerless to stop it.
        Keeping the incumbent's box makes the whole class impossible, because
        the track it is holding never leaves the list.

        Score only changes on a requery (every REQUERY_SECONDS), so even
        without an incumbent the winner is stable between them rather than
        swapping frame to frame the way a salience-only rule would; the
        salience tie-break decides only exact score ties, which floats make
        rare.

        The losers keep their labels and their tracking — they are simply not
        drawn, not attended to, and not written as sightings, which also stops
        one object logging a sighting per box. Case-folded, like every other
        label comparison here, for shards written before labels were
        lowercased.

        This deliberately means two identical objects in frame — two of the
        same mug — show one box between them. That is the accepted cost: the
        labels are the same string, so the robot has nothing to tell them apart
        with, and drawing the same name twice is the confusing half of it.
        """
        # last pass's attention, since this runs before focused() picks the
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

    def teach(self, crop, transcript, frame=None, box=None):
        """One spoken sentence → one point carrying BOTH named vectors.

        Takes the transcript, not the WAV: speech-to-text is slow, so the
        caller runs it before claiming the live-state lock.

        `frame`/`box` are optional and only decorate the memory: they save the
        wider scene for the memory tab to show. Recognition sees the crop and
        nothing else, exactly as before.

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
            scene=self._scene(frame, box),
        )
        self.log(f'taught "{label[:18]}" -> image + text')
        return {"id": pid, "label": label, "transcript": transcript}

    def confirm(self, label, frame=None):
        """One tap on the orange guess: teach the attending crop as that name.

        The maybe band shows "dylan? 0.87" — everything a teach needs is
        already on screen: the crop (the attending track's) and the name (the
        guess). Confirming skips the voice round-trip and its 6 s of Whisper.
        The transcript IS the label, which is the bare-noun teach recall's
        text search already handles ("Hat." scored 0.725 against the question
        that mattered).

        `label` is what the page displayed at tap time, verified against the
        live guess before acting — the FORGET lesson: focus moves under a
        finger already on its way down, and a tap meant for "dylan?" must not
        teach whatever the panel switched to. Returns the teach result, or
        None if the guess moved on (honest refusal, nothing written).

        The track is labeled in place — the point just written IS this crop,
        so the match is certain — which turns the box teal on the tap and
        makes a second tap a no-op instead of a duplicate view. requery_now
        still runs so score and thumb refresh at the next pass.
        """
        t = self.attention
        if (t is None or t.label or not t.guess or t.crop is None
                or t.guess.lower() != label.lower()):
            return None
        res = self.teach(t.crop, t.guess, frame=frame, box=t.box)
        t.label = res["label"]
        t.note = res["transcript"]
        t.guess = None
        t.requery_now()
        return res

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
            # case-folded to match memory.forget: a pre-lowercase shard can
            # have a track wearing the other casing of the same label
            if t.label and t.label.lower() == label.lower():
                t.label = t.note = t.thumb = None
                t.requery_now()
        for t in self.detector.tracks.values():
            # a guess wearing the deleted name would outlive it for a requery
            # beat otherwise — same reason the labels above are stripped
            if t.guess and t.guess.lower() == label.lower():
                t.guess = None
                t.requery_now()
        self.log(f'forgot "{label[:18]}" -> {n} point(s)')
        return n

    def rename(self, label, to):
        """Fix an object's name without touching what it looks like.

        The label is the key everything else matches on — recall dedupes by
        it, forget deletes by it — so it is lowercased here exactly as
        `audio.parse_label` lowercases what Whisper heard. Renaming onto a
        name that already exists deliberately MERGES the two: that is the
        cure for "Laptop" and "laptop" being two objects, and the memory tab
        groups by label, so it simply shows one card afterwards.
        """
        to = " ".join(to.split()).lower()
        n = self.memory.rename(label, to)
        for t in self.detector.tracks.values():
            # rename in place rather than requerying: the vectors did not
            # change, so the match is still the same match and the box can
            # just start saying the new name
            if t.label and t.label.lower() == label.lower():
                t.label = to
        self.log(f'renamed "{label[:14]}" -> "{to[:14]}" ({n} point(s))')
        return n

    def set_where(self, place):
        """Move the robot: change the place stamped on memories from now on.

        From the memory tab, so a relocated appliance doesn't need its
        service file edited (--location was the only way to set this).
        Old points keep their place — that is the feature, not a gap.
        """
        where = self.memory.set_where(place)
        self.log(f'here: "{where}"' if where else "location cleared")
        return where

    def forget_view(self, pid, label):
        """Delete ONE taught view — the cure for a single bad crop.

        Teaching an object again adds a view rather than replacing one, so a
        blurred or half-masked teach stays in memory forever, matching things
        it shouldn't. This removes just that one.

        If it is the object's last taught view, the whole object goes instead:
        deleting it alone would leave the label's sightings behind in recall,
        describing an object the robot can no longer recognize.

        Sightings are droppable through the same door now — the tab's cycle
        shows them, and a bad auto-saved photo (a stale box, a hand over the
        object) is as worth removing as a bad teach. A sighting drop deletes
        the whole BURST the shown row stands for (forget_sighting_burst):
        deleting only the shown point promotes a near-identical frame from
        seconds earlier into the same slot, which reads as a delete that did
        nothing — measured, not guessed. It never cascades to the object:
        only the last *taught* view takes the object with it.

        The point must actually belong to this object, and that is checked
        here rather than trusted: the id and the label arrive together from a
        page that may have been looking at a stale list, and two mistakes
        follow from believing it. A view already dropped from another tab
        would count as "the last one" and forget the whole object —
        sightings and all — on a press that asked to delete something that
        was already gone. And a pid belonging to something else (another
        object's view, an ignore point) would be deleted while the log said
        otherwise.
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
        """Dismiss an unknown so the robot stops offering it — clutter you
        don't want to teach. Persistent now: the dismissal writes the crop's
        vector to the shard (kind="ignored"), so the same-looking object is
        re-suppressed after a restart, when track ids mean nothing. It used
        to live only as a track id in the detector, and every power cut
        brought all the clutter back — reported as "ignored items get
        deleted each session".

        Takes the track rather than its id so a picture of what was dismissed
        can go with it: the memory tab lists these, and "IGNORED x3" with no
        pictures is not something an operator can undo with any confidence.
        """
        # The teach target has always been embedded (it is only offered after
        # its first query), so this fallback embed never runs in practice —
        # it is here so a caller with a never-queried track cannot write a
        # vectorless point.
        vec = track.vec if track.vec is not None else models.embed_crop(track.crop)
        pid = self.memory.ignore(vec, thumb=self._scene(frame, track.box))
        self.detector.ignore(track.tid, pid=pid)
        self.log("ignored one unknown")

    def unignore(self, pid):
        """Take one dismissal back, from the memory tab: delete its point and
        lift any live blocks that came from it."""
        ok = self.memory.unignore(pid)
        if ok:
            self.detector.unblock(pid)
        return ok

    # -- ask -------------------------------------------------------------------

    def ask(self, question, since_ts=None):
        """Recall: pick the object by what was SAID about it, then answer
        with where it was last seen.

        Two steps on purpose. The question and the taught transcripts live
        in the same text space, so Nomic picks the object reliably (0.725
        vs 0.424 for the runner-up, on the live shard). The old single-step
        version searched the image space with CLIP's text tower and the
        whole question — which put every sighting of the day in one flat
        0.246-0.262 band, so the spoken answer was noise wearing whatever
        label had the most sightings. And once the object is chosen, WHERE
        it was is a question about time, not similarity: the newest
        sightings, not the nearest vectors.

        "What did you see today?" is an inventory, not a search: the
        distinct objects sighted since morning, no vectors involved.
        ponytail: one phrase check, not intent parsing — the demo script
        asks this exact question. `since_ts` only scopes the inventory;
        "where did I leave it" deliberately looks back past midnight.
        """
        if "what did you see" in question.lower():
            since = day_start_ts() if since_ts is None else since_ts
            return {"inventory": True, "label": None, "note": None,
                    "score": 0.0, "sightings": self.memory.seen_since(since)}
        hit = self.memory.best_taught(models.embed_query(question))
        if hit is None:
            return {"inventory": False, "label": None, "note": None,
                    "score": 0.0, "sightings": []}
        label = hit.payload.get("label")
        return {"inventory": False, "label": label,
                "note": hit.payload.get("transcript"),
                "score": hit.score,
                "sightings": self.memory.last_sightings(label)}

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
