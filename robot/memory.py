"""The robot's memory: one Qdrant Edge shard, two named vectors.

Same shape as the course's L5 assistant_shard — text 768 (Nomic), image 512
(CLIP), cosine — and the same nearest-match-vs-threshold recognition check.
The threshold itself is per-camera and lives in .env; see RECOGNIZE_THRESHOLD.
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

# Per-camera, so it lives in .env (see robot/config.py) and is overridable
# per-run with --threshold. L5 teaches 0.80 against clean studio crops; a live
# camera's crops are smaller and softer, which lifts every score, so the
# calibrated default here is higher. Re-run testdata/verify_scores.py after
# any change to the encoders or the crop pipeline — it prints the knee.
RECOGNIZE_THRESHOLD = config.RECOGNIZE_THRESHOLD

CONFIG = EdgeConfig(
    vectors={
        "text": EdgeVectorParams(size=768, distance=Distance.Cosine),
        "image": EdgeVectorParams(size=512, distance=Distance.Cosine),
    }
)


class Memory:
    """Store, recognize, recall — the L2–L5 lifecycle behind the robot."""

    def __init__(self, data_dir, threshold=RECOGNIZE_THRESHOLD, where=None):
        self.threshold = threshold
        self.dir = Path(data_dir)
        # Serializes every shard access, including reopen/close. This is what
        # lets the memory tab read the shard WITHOUT the app's live-state
        # lock: that lock is held by the detect thread for a whole YOLO pass,
        # so a tab page used to wait 0.8-1.6 s (measured) for a scan that
        # costs half a millisecond. An RLock because compound reads
        # (objects -> count_label) nest.
        self._lock = threading.RLock()
        # The place stamped on every write (memory-fleet: payload "where").
        # A --location flag wins for the run; otherwise the file the memory
        # tab's SET LOCATION button writes. Beside the shard, not in .env:
        # it is state that travels with the memories, not calibration.
        self._where_file = self.dir / "where.txt"
        if where is None and self._where_file.exists():
            where = self._where_file.read_text().strip() or None
        self.where = where
        # A shard is present if its segments are. The glob keeps an
        # unrecognized non-empty dir failing loud in load() rather than
        # being silently built over — but the two things this app itself
        # puts beside the shard (thumbs/, where.txt) must not count, or a
        # dir holding only them (segments hand-deleted, a restore that
        # died early) turns into a load() of nothing that crashes every
        # boot. Caught by review after where.txt became the second one.
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
        """Set (or clear) the place stamped on writes from now on — existing
        points keep the place they were written at, which is the point:
        "where are my keys" should answer where it saw them.

        Persisted beside the shard, so the appliance keeps it across the
        power cut that is its documented off switch — a location that
        silently reverted to the service file's idea of home would stamp a
        booth's memories with the living room. Normalized and capped like a
        rename, since recall reads it out loud.
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
            # durable now: the offline-reboot beat power-cycles the device, so
            # a write buffered in the WAL until a clean close() would be lost.
            self.shard.flush()
            return pid

    def teach(self, image_vec, text_vec, label, transcript, ts=None, thumb=None,
              scene=None):
        """One point, BOTH named vectors — searchable by sight and by words.

        ponytail: re-teaching the same object adds a second point; recognition
        returns whichever view is nearest, so the newer note isn't guaranteed
        to win. Fold-by-label (as in memory-fleet) if that ever matters.

        `scene` is the object as the camera saw it — the detector's box plus
        a margin, unmasked — and is shown instead of the crop in the memory
        tab, because a masked gray-filled crop is what CLIP compares, not what
        a human recognizes. Points written before it existed simply have no
        scene, and the tab falls back to the crop. (Points written during one
        brief window hold a whole frame instead; same fallback, nothing to
        migrate.)
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
        """Delete every point for a label — taught views and sightings alike.

        The L2 forget beat is `delete_points([id])`; here we gather all ids
        for the label first so re-taught views (see `teach`) are removed too,
        not just the matched one.
        """
        from qdrant_edge import ScrollRequest
        with self._lock:
            # ponytail: full scan, fine at demo scale (tens–hundreds of points)
            records, _ = self.shard.scroll(
                ScrollRequest(limit=10000, with_payload=True))
            # case-folded so a shard written before labels were lowercased
            # still forgets whole objects: "Water bottle" and "water bottle"
            # are one thing, and deleting only the matching-case half leaves a
            # green box that will not die — which reads as a broken FORGET
            ids = [p.id for p in records
                   if (p.payload.get("label") or "").lower() == label.lower()]
            if ids:
                self.shard.update(UpdateOperation.delete_points(ids))
                self.shard.flush()
            return len(ids)

    def rename(self, label, new_label):
        """Give every point wearing one label a different one.

        Whisper mishears a bare noun ("laptop" comes back as "La-caw"), and
        until this existed the only cure was to forget the object and teach it
        again — throwing away vectors that were perfectly good, because the
        *word* was wrong. Nothing here touches a vector.

        `set_payload`, not a re-upsert: `UpdateOperation.upsert_points` with an
        existing id does NOT rewrite the point in this qdrant_edge build — it
        reports nothing and changes nothing. Other payload keys survive.
        """
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(
                ScrollRequest(limit=10000, with_payload=True))
            # case-folded, like forget: a pre-lowercase shard can hold both
            # spellings of one object and a rename must move all of it
            ids = [p.id for p in records
                   if (p.payload.get("label") or "").lower() == label.lower()]
            if ids:
                self.shard.update(
                    UpdateOperation.set_payload(ids, {"label": new_label}))
                self.shard.flush()
            return len(ids)

    def taught_ids(self, label):
        """Point ids of every taught view wearing this label, case-folded."""
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000, with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="taught")),
                ])))
            return [p.id for p in records
                    if (p.payload.get("label") or "").lower() == label.lower()]

    def forget_point(self, pid):
        """Delete exactly one point — one taught view, not the whole object."""
        with self._lock:
            self.shard.update(UpdateOperation.delete_points([pid]))
            self.shard.flush()

    def remember_sighting(self, image_vec, label, ts=None, thumb=None):
        """A cadence write: something stable in view, image vector only."""
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

    # -- reads ----------------------------------------------------------------

    def count_label(self, label, kind="seen"):
        """How many points of one kind wear this label — the tab's sighting
        count. Exact-case: labels are lowercased at parse time (see
        audio.parse_label), so only a shard written before that change can
        undercount, and only in a number shown for interest."""
        from qdrant_edge import CountRequest
        with self._lock:
            return self.shard.count(CountRequest(
                exact=True, filter=Filter(must=[
                    FieldCondition(key="kind", match=MatchValue(value=kind)),
                    FieldCondition(key="label", match=MatchValue(value=label)),
                ])))

    def objects(self):
        """Every taught object, newest first: one entry per label, all its views.

        This is what the memory tab lists — the answer to "what do you know?",
        which used to be answerable only by pointing the camera at things one
        at a time and reading the box.

        Grouped in Python rather than by a facet query, and case-folded the
        same way `forget` is, so the tab's idea of one object is exactly the
        tab's delete button's idea of one object. A full
        scan of the taught points: there are single digits of them at demo
        scale, and the label is not indexed.
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
                # display the newest view's spelling, since that is the one a
                # re-teach just wrote
                g["label"] = g["views"][0].get("label") or g["label"]
                g["seen"] = self.count_label(g["label"])
                out.append(g)
            out.sort(key=lambda g: g["views"][0].get("ts") or 0, reverse=True)
            return out

    def recognize(self, image_vec):
        """Nearest taught view vs the threshold. Returns (hit|None, score)."""
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
            return None, 0.0
        top = hits[0]
        if top.score >= self.threshold:
            return top, top.score
        return None, top.score

    def best_taught(self, text_vec):
        """The taught object whose TRANSCRIPT best matches a question —
        recall's first step.

        Text space on purpose. "Where did I leave my hat?" against "Hat."
        scores 0.725 with the runner-up at 0.424 (measured on the live
        shard), because the question and the transcript are both language.
        The CLIP text-to-image search this replaced scored every sighting of
        the day in one flat 0.246-0.262 band — the top answer was noise
        wearing whatever label had the most sightings. See "Recall answered
        from the wrong space" in CLAUDE.md before bringing that back.
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

    def last_sightings(self, label, limit=3, apart=600):
        """The newest sightings of one object, newest first and at least
        `apart` seconds between rows — where it was, as distinct occasions.

        A scroll and a sort, not a vector search: once the object is chosen,
        "where did I leave it" is a question about time, and the
        nearest-vector sighting can be a week old. Deliberately NOT
        day-filtered — keys left yesterday are the whole use case.
        Case-folded like every other label comparison here.

        The `apart` collapse is load-bearing, not cosmetic: a sighting is
        written per track birth, and an object sitting in view births tracks
        every few minutes, so the raw newest three are one moment three
        times over — "at 10:42 PM, before that at 10:42 PM and 10:42 PM",
        observed on the live shard. Keeping the newest of each burst is what
        makes the second and third rows carry any information.
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
                        - (r.payload.get("ts") or 0)) < apart:
                continue
            out.append(r)
            if len(out) >= limit:
                break
        return out

    def seen_since(self, since_ts, limit=4):
        """Distinct objects sighted since a time, one row each (the newest) —
        the day-inventory answer to "what did you see today?". No vectors:
        an inventory is a time filter, not a search."""
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
