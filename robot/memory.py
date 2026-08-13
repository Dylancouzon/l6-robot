"""The robot's memory: one Qdrant Edge shard, two named vectors.

Same shape as the course's L5 assistant_shard — text 768 (Nomic), image 512
(CLIP), cosine — and the same nearest-match-vs-threshold recognition check.
The threshold itself is per-camera and lives in .env; see RECOGNIZE_THRESHOLD.
"""
import itertools
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


def _dedupe_by_label(hits):
    """One entry per label — the nearest view wins (hits arrive score-sorted).

    Re-teaching an object stores extra views (see `teach`), so recall can
    otherwise list the same label several times. Case-folded: new labels are
    lowercased at parse time, but a shard written before that change can
    still hold "Water bottle" next to "water bottle" — one object, not two.
    """
    seen, out = set(), []
    for h in hits:
        label = h.payload.get("label")
        key = label.lower() if label else label
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


class Memory:
    """Store, recognize, recall — the L2–L5 lifecycle behind the robot."""

    def __init__(self, data_dir, threshold=RECOGNIZE_THRESHOLD, where=None):
        self.threshold = threshold
        self.where = where  # place stamped on every write this session (memory-fleet: payload "where")
        self.dir = Path(data_dir)
        if (self.dir / "segments").exists() or any(self.dir.glob("*")):
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
        self.shard.close()

    def reopen(self):
        """The offline-reboot beat: flush, drop the handle, reload from disk."""
        self.shard.close()
        self.shard = EdgeShard.load(str(self.dir))

    def count(self):
        from qdrant_edge import CountRequest
        return self.shard.count(CountRequest(exact=True))

    # -- writes ---------------------------------------------------------------

    def _upsert(self, vector, payload):
        pid = next(self._ids)
        self.shard.update(UpdateOperation.upsert_points([
            Point(id=pid, vector=vector, payload=payload)
        ]))
        # durable now: the offline-reboot beat power-cycles the device, so a
        # write buffered in the WAL until a clean close() would be lost.
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
        # ponytail: full scan, fine at demo scale (tens–hundreds of points)
        records, _ = self.shard.scroll(
            ScrollRequest(limit=10000, with_payload=True))
        # case-folded so a shard written before labels were lowercased still
        # forgets whole objects: "Water bottle" and "water bottle" are one
        # thing, and deleting only the matching-case half leaves a green box
        # that will not die — which reads as a broken FORGET
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
        records, _ = self.shard.scroll(ScrollRequest(
            limit=10000, with_payload=True,
            filter=Filter(must=[
                FieldCondition(key="kind", match=MatchValue(value="taught")),
            ])))
        return [p.id for p in records
                if (p.payload.get("label") or "").lower() == label.lower()]

    def forget_point(self, pid):
        """Delete exactly one point — one taught view, not the whole object."""
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
        return self.shard.count(CountRequest(exact=True, filter=Filter(must=[
            FieldCondition(key="kind", match=MatchValue(value=kind)),
            FieldCondition(key="label", match=MatchValue(value=label)),
        ])))

    def objects(self):
        """Every taught object, newest first: one entry per label, all its views.

        This is what the memory tab lists — the answer to "what do you know?",
        which used to be answerable only by pointing the camera at things one
        at a time and reading the box.

        Grouped in Python rather than by a facet query, and case-folded the
        same way `forget` and `_dedupe_by_label` are, so the tab's idea of one
        object is exactly the tab's delete button's idea of one object. A full
        scan of the taught points: there are single digits of them at demo
        scale, and the label is not indexed.
        """
        from qdrant_edge import ScrollRequest
        records, _ = self.shard.scroll(ScrollRequest(
            limit=10000, with_payload=True,
            filter=Filter(must=[
                FieldCondition(key="kind", match=MatchValue(value="taught")),
            ])))
        groups = {}
        for r in records:
            label = r.payload.get("label") or ""
            g = groups.setdefault(label.lower(), {"label": label, "views": []})
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
        hits = self.shard.query(QueryRequest(
            query=Query.Nearest(image_vec, using="image"),
            filter=Filter(must=[
                FieldCondition(key="kind", match=MatchValue(value="taught")),
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

    def day_recall(self, text_vec, clip_text_vec, since_ts, limit=4):
        """'What did you see today?' — both spaces, time-filtered, never merged."""
        window = Filter(must=[
            FieldCondition(key="ts", range=RangeFloat(gte=since_ts)),
        ])
        # over-fetch so dedupe still leaves ~limit distinct objects
        heard = self.shard.query(QueryRequest(
            query=Query.Nearest(text_vec, using="text"),
            filter=window,
            limit=limit * 3,
            with_payload=True,
        ))
        seen = self.shard.query(QueryRequest(
            query=Query.Nearest(clip_text_vec, using="image"),
            filter=window,
            limit=limit * 3,
            with_payload=True,
        ))
        return {"seen": _dedupe_by_label(seen)[:limit],
                "heard": _dedupe_by_label(heard)[:limit]}
