"""The robot's memory: one Qdrant Edge shard, two named vectors.

Same shape as the course's L5 assistant shard - `text` 768 (Nomic), `image`
512 (CLIP), cosine - and the same nearest-match-against-a-threshold check.

Every point carries `kind`, which is the whole taxonomy:

    taught    an object someone named out loud. Both vectors.
    seen      the robot noticed a taught object. Image vector only.
    ignored   clutter someone dismissed. Image vector only.
"""
import itertools
import threading
import time
from pathlib import Path

from robot import config

from qdrant_edge import (
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    Point,
    Query,
    QueryRequest,
    RangeFloat,
    UpdateOperation,
)

# Per-camera, so it lives in .env and is overridable per run with --threshold.
# Re-run testdata/verify_scores.py after any change to the encoders or crops.
RECOGNIZE_THRESHOLD = config.RECOGNIZE_THRESHOLD

# The "maybe" band under the bar: a nearest-taught score within this margin is
# surfaced as a guess - an orange "hat?" box - instead of a bare UNKNOWN.
# Display only. Nothing recognizes, logs a sighting or can be forgotten off a
# guess, which is what makes showing the band safe when lowering the bar is not.
MAYBE_MARGIN = 0.05

# One "occasion" of being seen: sightings of a label closer together than this
# are one burst, shown as one row and deleted as one unit. Named once because
# those two must agree - a display window wider than the delete window
# resurrects rows that were just deleted.
SIGHTING_APART = 600

CONFIG = EdgeConfig(
    vectors={
        "text": EdgeVectorParams(size=768, distance=Distance.Cosine),
        "image": EdgeVectorParams(size=512, distance=Distance.Cosine),
    }
)


class Memory:
    """Store, recognize, recall - the L2-L5 lifecycle behind the robot."""

    def __init__(self, data_dir, threshold=RECOGNIZE_THRESHOLD, where=None):
        self.threshold = threshold
        self.dir = Path(data_dir)
        # Serializes every shard access, including reopen and close. This is
        # what lets the memory tab read the shard without the app's live-state
        # lock, which the detect thread holds for a whole detection pass.
        # An RLock because compound reads nest (objects -> count_sightings).
        self._lock = threading.RLock()
        # The place stamped on every write. --location wins for the run;
        # otherwise the file the memory tab's SET LOCATION button writes. It
        # lives beside the shard because it is state that travels with the
        # memories, not per-camera calibration.
        self._where_file = self.dir / "where.txt"
        if where is None and self._where_file.exists():
            where = self._where_file.read_text().strip() or None
        self.where = where
        # A shard is present if its segments are. The glob makes an
        # unrecognized non-empty directory fail loudly in load() rather than
        # being silently built over - but the two things this app puts beside
        # the shard must not count, or a directory holding only them turns into
        # a load() of nothing that crashes every boot.
        others = [p for p in self.dir.glob("*")
                  if p.name not in ("thumbs", "where.txt")]
        if (self.dir / "segments").exists() or others:
            self.shard = EdgeShard.load(str(self.dir))
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.shard = EdgeShard.create(str(self.dir), CONFIG)
            for key, schema in [
                ("kind", PayloadSchemaType.Keyword),
                ("ts", PayloadSchemaType.Float),
            ]:
                self.shard.update(
                    UpdateOperation.create_field_index(key, schema)
                )
        # Ids are seeded from a nanosecond clock so they stay unique across
        # restarts without scanning the shard for the highest one. They land
        # around 1.8e18, which is past JavaScript's safe integer range - see
        # LiveApp._view_json, which sends them to the page as strings.
        self._ids = itertools.count(time.time_ns())

    def close(self):
        with self._lock:
            self.shard.close()

    def reopen(self):
        """The offline-reboot beat: flush, drop the handle, reload from disk."""
        with self._lock:
            self.shard.close()
            self.shard = EdgeShard.load(str(self.dir))

    def set_where(self, place):
        """Set or clear the place stamped on writes from now on.

        Existing points keep the place they were written at, which is the
        point: "where are my keys" should answer where it saw them. Persisted
        beside the shard so it survives the power cut that is the appliance's
        off switch.
        """
        place = " ".join((place or "").split())[:60] or None
        with self._lock:
            self.where = place
            self._where_file.write_text(place or "")
        return place

    def count(self):
        from qdrant_edge import CountRequest
        with self._lock:
            return self.shard.count(CountRequest(exact=True))

    # -- writes ---------------------------------------------------------------

    def _upsert(self, vector, payload):
        with self._lock:
            pid = next(self._ids)
            self.shard.update(UpdateOperation.upsert_points([
                Point(id=pid, vector=vector, payload=payload)
            ]))
            # Flushed immediately: pulling the plug is the documented off
            # switch, so nothing may sit in the WAL waiting for a clean close.
            self.shard.flush()
            return pid

    def teach(self, image_vec, text_vec, label, transcript, ts=None, thumb=None,
              scene=None):
        """One point carrying both named vectors: searchable by sight and words.

        `thumb` is the masked crop recognition compares. `scene` is the object
        as the camera saw it - the box plus a margin, unmasked - which is what
        the memory tab shows, because a gray masked cut-out is not something a
        human can identify their own object from.

        Re-teaching the same object adds a second point rather than replacing
        one. That is deliberate: recognition matches the nearest view, so more
        views is how an object gets recognized from more angles.
        """
        return self._upsert(
            {"image": image_vec, "text": text_vec},
            {
                "kind": "taught",
                "label": label,
                "transcript": transcript,
                "ts": ts or time.time(),
                "thumb": thumb,
                "scene": scene,
                "where": self.where,
            },
        )

    def forget(self, label):
        """Delete every point for a label, taught views and sightings alike.

        Case-folded, like every label comparison here, so a shard written
        before labels were lowercased still forgets whole objects.
        """
        from qdrant_edge import ScrollRequest
        with self._lock:
            # ponytail: full scan, fine at demo scale (tens-hundreds of points)
            records, _ = self.shard.scroll(
                ScrollRequest(limit=10000, with_payload=True))
            ids = [p.id for p in records
                   if (p.payload.get("label") or "").lower() == label.lower()]
            if ids:
                self.shard.update(UpdateOperation.delete_points(ids))
                self.shard.flush()
            return len(ids)

    def rename(self, label, new_label):
        """Give every point wearing one label a different one. No vector moves.

        The cure for Whisper mishearing a bare noun: the views were fine, only
        the word was wrong.

        set_payload, not a re-upsert: UpdateOperation.upsert_points with an
        existing id does NOT rewrite the point in this qdrant_edge build.
        """
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(
                ScrollRequest(limit=10000, with_payload=True))
            ids = [p.id for p in records
                   if (p.payload.get("label") or "").lower() == label.lower()]
            if ids:
                self.shard.update(
                    UpdateOperation.set_payload(ids, {"label": new_label}))
                self.shard.flush()
            return len(ids)

    def _ids_of_kind(self, kind, label):
        """Point ids of one kind wearing this label, case-folded."""
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000, with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind", match=MatchValue(value=kind)),
                ])))
            return [p.id for p in records
                    if (p.payload.get("label") or "").lower() == label.lower()]

    def taught_ids(self, label):
        """Every taught view of this object."""
        return self._ids_of_kind("taught", label)

    def seen_ids(self, label):
        """Every sighting of this object, so one photo can be dropped."""
        return self._ids_of_kind("seen", label)

    def forget_point(self, pid):
        """Delete exactly one point: one taught view, not the whole object."""
        with self._lock:
            self.shard.update(UpdateOperation.delete_points([pid]))
            self.shard.flush()

    def remember_sighting(self, image_vec, label, ts=None, thumb=None):
        """A cadence write: something known is in view. Image vector only."""
        return self._upsert(
            {"image": image_vec},
            {
                "kind": "seen",
                "label": label,
                "ts": ts or time.time(),
                "thumb": thumb,
                "where": self.where,
            },
        )

    def ignore(self, image_vec, thumb=None, ts=None):
        """Persist one dismissal: what the ignored crop looked like.

        This is what makes IGNORE survive a restart. Track ids restart from 1
        every run, so the vector is the only durable name the object has.
        """
        return self._upsert(
            {"image": image_vec},
            {
                "kind": "ignored",
                "ts": ts or time.time(),
                "thumb": thumb,
                "where": self.where,
            },
        )

    def unignore(self, pid):
        """Delete one ignored point, verified against the shard first.

        The id arrives from a page that may be stale, and an id belonging to
        something else must not be deletable through this door.
        """
        with self._lock:
            if pid not in [r.id for r in self.ignored()]:
                return False
            self.shard.update(UpdateOperation.delete_points([pid]))
            self.shard.flush()
            return True

    def ignored(self):
        """Every persisted dismissal, newest first - the tab's IGNORED list."""
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000, with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="ignored")),
                ])))
        records.sort(key=lambda r: r.payload.get("ts") or 0, reverse=True)
        return records

    def match_ignored(self, image_vec):
        """Nearest ignored crop against the same threshold recognition uses.

        Only consulted for tracks that did NOT match a taught object, so a
        taught label always beats an ignore: a degenerate ignored crop can at
        worst hide unknown clutter, never something deliberately taught.
        """
        with self._lock:
            hits = self.shard.query(QueryRequest(
                query=Query.Nearest(image_vec, using="image"),
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="ignored")),
                ]),
                limit=1,
                with_payload=True,
            ))
        if hits and hits[0].score >= self.threshold:
            return hits[0]
        return None

    # -- reads ----------------------------------------------------------------

    def count_sightings(self, label):
        """How many times this object has been seen - the tab's count."""
        from qdrant_edge import CountRequest
        with self._lock:
            return self.shard.count(CountRequest(
                exact=True, filter=Filter(must=[
                    FieldCondition(key="kind", match=MatchValue(value="seen")),
                    FieldCondition(key="label", match=MatchValue(value=label)),
                ])))

    def objects(self):
        """Every taught object, newest first: one entry per label, all its views.

        What the memory tab lists, and the answer to "what do you know?".
        Grouped in Python and case-folded the same way forget is, so the tab's
        idea of one object matches its delete button's idea of one object.
        """
        from qdrant_edge import ScrollRequest
        # the whole grouping under one lock hold, so a REBOOT cannot swap the
        # shard out between the scroll and the per-label counts
        with self._lock:
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000, with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="taught")),
                ])))
            groups = {}
            for r in records:
                label = r.payload.get("label") or ""
                g = groups.setdefault(label.lower(),
                                      {"label": label, "views": []})
                # the id rides along so the tab can drop one bad view without
                # taking the object's good ones with it
                g["views"].append(dict(r.payload, id=r.id))
            out = []
            for g in groups.values():
                g["views"].sort(key=lambda p: p.get("ts") or 0, reverse=True)
                # display the newest view's spelling, since a re-teach wrote it
                g["label"] = g["views"][0].get("label") or g["label"]
                g["seen"] = self.count_sightings(g["label"])
                # The robot's own photos, which recall answers with. Uncapped
                # on purpose: this list is what DROP works through, and behind
                # a cap deleting one row just slides the next one in, which
                # reads as a delete that did nothing.
                g["sightings"] = [dict(r.payload, id=r.id) for r in
                                  self.last_sightings(g["label"], limit=None)]
                out.append(g)
            out.sort(key=lambda g: g["views"][0].get("ts") or 0, reverse=True)
            return out

    def recognize(self, image_vec):
        """Nearest taught view against the threshold. This is the whole rule.

        Returns (hit|None, score, guess). `guess` is the nearest taught label
        when the score lands just under the bar, and is for display only: the
        caller must not treat a guess as recognition.
        """
        with self._lock:
            hits = self.shard.query(QueryRequest(
                query=Query.Nearest(image_vec, using="image"),
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="taught")),
                ]),
                limit=1,
                with_payload=True,
            ))
        if not hits:
            return None, 0.0, None
        top = hits[0]
        if top.score >= self.threshold:
            return top, top.score, None
        if top.score >= self.threshold - MAYBE_MARGIN:
            return None, top.score, top.payload.get("label")
        return None, top.score, None

    def best_taught(self, text_vec):
        """The taught object whose transcript best matches a question.

        Recall's first step, and it searches the TEXT space: the question and
        the transcript are both language, so they land near each other. See
        "Recall" in CLAUDE.md for what searching the image space did instead.
        """
        with self._lock:
            hits = self.shard.query(QueryRequest(
                query=Query.Nearest(text_vec, using="text"),
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="taught")),
                ]),
                limit=1,
                with_payload=True,
            ))
        return hits[0] if hits else None

    def forget_sighting_burst(self, pid):
        """Delete one sighting and the burst it stands for. Returns how many.

        Rows are burst-collapsed everywhere they are shown, so deleting only
        the tapped point does nothing visible: the next near-identical frame
        slides into its slot. The unit of deletion has to be the unit of
        display. The label comes from the point itself, never the caller.
        """
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000, with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="seen")),
                ])))
            target = next((r for r in records if r.id == pid), None)
            if target is None:
                return 0
            label = (target.payload.get("label") or "").lower()
            ts = target.payload.get("ts") or 0
            ids = [r.id for r in records
                   if (r.payload.get("label") or "").lower() == label
                   and ts - SIGHTING_APART < (r.payload.get("ts") or 0) <= ts]
            self.shard.update(UpdateOperation.delete_points(ids))
            self.shard.flush()
            return len(ids)

    def last_sightings(self, label, limit=3):
        """The newest sightings of one object, as distinct occasions.

        A scroll and a sort, not a vector search: once the object is chosen,
        "where did I leave it" is a question about time. Deliberately not
        day-filtered - keys left yesterday are the whole use case.

        The burst collapse is what makes the second and third rows carry any
        information: a sighting is written per track birth, so an object
        sitting in view produces the same moment over and over.
        """
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000, with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="seen")),
                ])))
        rows = [r for r in records
                if (r.payload.get("label") or "").lower() == label.lower()]
        rows.sort(key=lambda r: r.payload.get("ts") or 0, reverse=True)
        out = []
        for r in rows:
            if out and ((out[-1].payload.get("ts") or 0)
                        - (r.payload.get("ts") or 0)) < SIGHTING_APART:
                continue
            out.append(r)
            if limit is not None and len(out) >= limit:
                break
        return out

    def seen_since(self, since_ts, limit=4):
        """Distinct objects sighted since a time, newest row each.

        The answer to "what did you see today?": an inventory is a time
        filter, not a search, so no vectors are involved.
        """
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000, with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="seen")),
                    FieldCondition(key="ts",
                                   range=RangeFloat(gte=since_ts)),
                ])))
        records.sort(key=lambda r: r.payload.get("ts") or 0, reverse=True)
        out, seen = [], set()
        for r in records:
            key = (r.payload.get("label") or "").lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
            if len(out) >= limit:
                break
        return out
