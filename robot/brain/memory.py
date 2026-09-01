"""The robot's memory: one Qdrant Edge shard, two named vectors.

`text` 768 (Nomic), `image` 512 (CLIP), cosine, and a
nearest-match-against-a-threshold check.

Every point carries `kind`, which is the whole taxonomy:

    taught    an object someone named out loud. Both vectors.
    seen      the robot noticed a taught object. Image vector only.
    ignored   clutter someone dismissed. Image vector only.
"""
import itertools
import re
import threading
import time
from pathlib import Path

from qdrant_edge import (
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    Point,
    PointVectors,
    Query,
    QueryRequest,
    RangeFloat,
    UpdateOperation,
)

from robot import config

# Per-camera, so it lives in .env and is overridable per run with --threshold.
# Re-run testdata/verify_scores.py after any change to the encoders or crops.
RECOGNIZE_THRESHOLD = config.RECOGNIZE_THRESHOLD

# Scores just below the threshold appear as guesses in the UI. They never
# count as recognition or create sightings.
MAYBE_MARGIN = 0.05

# Group nearby sightings into one occasion for both display and deletion.
SIGHTING_APART = 600

CONFIG = EdgeConfig(
    vectors={
        "text": EdgeVectorParams(size=768, distance=Distance.Cosine),
        "image": EdgeVectorParams(size=512, distance=Distance.Cosine),
    }
)


class Memory:
    """Store taught views, recognize objects, and recall their sightings."""

    def __init__(self, data_dir, threshold=RECOGNIZE_THRESHOLD, where=None):
        self.threshold = threshold
        self.dir = Path(data_dir)
        # Compound reads call other locked methods, so this must be an RLock.
        self._lock = threading.RLock()
        # Location travels with the memories rather than camera calibration.
        self._where_file = self.dir / "where.txt"
        if where is None and self._where_file.exists():
            where = self._where_file.read_text(encoding="utf-8").strip() or None
        self.where = where
        # Load any existing shard directory, but ignore files this app stores
        # alongside a shard when deciding whether one exists.
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
        # Clock-based IDs avoid a shard scan. The UI sends them as strings
        # because they exceed JavaScript's safe integer range.
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
        """Set the place on future writes without changing existing memories."""
        place = " ".join((place or "").split())[:60] or None
        with self._lock:
            self.where = place
            self._where_file.write_text(place or "", encoding="utf-8")
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
            # The appliance may lose power without a clean shutdown.
            self.shard.flush()
            return pid

    def teach(self, image_vec, text_vec, label, transcript, ts=None, thumb=None,
              scene=None):
        """Store a view with image and text vectors.

        Re-teaching adds another view so recognition can match more angles.
        `thumb` is the masked recognition crop; `scene` is the UI image.
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
        """Delete all taught views and sightings for a label, case-insensitively."""
        from qdrant_edge import ScrollRequest
        with self._lock:
            # A full scan is acceptable at the robot's small scale.
            records, _ = self.shard.scroll(
                ScrollRequest(limit=10000, with_payload=True))
            ids = [p.id for p in records
                   if (p.payload.get("label") or "").lower() == label.lower()]
            if ids:
                self.shard.update(UpdateOperation.delete_points(ids))
                self.shard.flush()
            return len(ids)

    def rename(self, label, new_label, text_vec=None):
        """Rename an object and optionally replace its taught text vectors.

        Sightings have no text vector. Existing points are updated in place
        because this Qdrant Edge version does not replace them by upserting.
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
                if text_vec is not None:
                    moved = set(ids)
                    taught = [p.id for p in records
                              if p.id in moved
                              and p.payload.get("kind") == "taught"]
                    if taught:
                        self.shard.update(UpdateOperation.update_vectors(
                            [PointVectors(pid, {"text": text_vec})
                             for pid in taught]))
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
        """Store an ignored crop so the dismissal survives a restart."""
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
        """Delete an ignored point after checking its kind."""
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
        """Find an ignored crop using the recognition threshold."""
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
        """Return taught objects newest first, grouped case-insensitively."""
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
                # Include the ID so the UI can delete one view.
                g["views"].append(dict(r.payload, id=r.id))
            out = []
            for g in groups.values():
                g["views"].sort(key=lambda p: p.get("ts") or 0, reverse=True)
                # Use the capitalization from the newest view.
                g["label"] = g["views"][0].get("label") or g["label"]
                g["seen"] = self.count_sightings(g["label"])
                # Keep all sightings available because each can be deleted.
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

    def names_in(self, question):
        """Return taught labels that appear as words in a question.

        Matching ignores case and a trailing plural "s" on longer words.
        One-token labels shorter than three characters are skipped to avoid
        common transcription noise. The caller uses every match as a candidate
        for semantic search.
        """
        from qdrant_edge import ScrollRequest

        def toks(text):
            out = []
            for w in re.split(r"[^0-9a-z]+", text.lower()):
                if w:
                    out.append(w[:-1] if len(w) >= 4 and w.endswith("s") else w)
            return out

        asked = toks(question)
        with self._lock:
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000, with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="taught")),
                ])))
        hits = {}
        for r in records:
            label = r.payload.get("label") or ""
            want = toks(label)
            if not want or (len(want) == 1 and len(want[0]) < 3):
                continue
            if any(asked[i:i + len(want)] == want
                   for i in range(len(asked) - len(want) + 1)):
                ts = r.payload.get("ts") or 0
                hits[label] = max(ts, hits.get(label, 0))
        return [l for l, _ in sorted(hits.items(), key=lambda kv: -kv[1])]

    def best_taught(self, text_vec, labels=None):
        """Find the taught transcript closest to a question vector.

        When `labels` is nonempty, only those explicitly named objects are
        candidates. This score is displayed but has no acceptance threshold.
        """
        must = [FieldCondition(key="kind", match=MatchValue(value="taught"))]
        if labels:
            must.append(
                FieldCondition(key="label", match=MatchAny(list(labels))))
        with self._lock:
            hits = self.shard.query(QueryRequest(
                query=Query.Nearest(text_vec, using="text"),
                filter=Filter(must=must),
                limit=1,
                with_payload=True,
            ))
        return hits[0] if hits else None

    def forget_sighting_burst(self, pid):
        """Delete the sighting occasion represented by one point."""
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

    def last_sightings(self, label, limit=3, kinds=("seen",)):
        """Return the newest sightings, grouping nearby writes into occasions.

        Recall includes taught points because teaching also records a known
        time and place. The memory tab uses the default of sightings only.
        """
        from qdrant_edge import ScrollRequest
        with self._lock:
            # Filter by kind before applying the scroll limit.
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000, with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind", match=MatchAny(list(kinds))),
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
        """Return the latest sighting of each object since a timestamp."""
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
