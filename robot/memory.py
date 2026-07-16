"""The robot's memory: one Qdrant Edge shard, two named vectors.

Same shape as the course's L5 assistant_shard — text 768 (Nomic), image 512
(CLIP), cosine — and the same nearest-match-vs-0.80 recognition check.
"""
import itertools
import time
from pathlib import Path

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

# Default matches L5. Calibratable (--threshold): demo quality wins; if live
# calibration moves it, L5 gets edited to match so "same number" stays true.
RECOGNIZE_THRESHOLD = 0.80

CONFIG = EdgeConfig(
    vectors={
        "text": EdgeVectorParams(size=768, distance=Distance.Cosine),
        "image": EdgeVectorParams(size=512, distance=Distance.Cosine),
    }
)


class Memory:
    """Store, recognize, recall — the L2–L5 lifecycle behind the robot."""

    def __init__(self, data_dir, threshold=RECOGNIZE_THRESHOLD):
        self.threshold = threshold
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
        return pid

    def teach(self, image_vec, text_vec, label, transcript, ts=None, thumb=None):
        """One point, BOTH named vectors — searchable by sight and by words.

        ponytail: re-teaching the same object adds a second point; recognition
        returns whichever view is nearest, so the newer note isn't guaranteed
        to win. Fold-by-label (as in memory-fleet) if that ever matters.
        """
        return self._upsert(
            {"image": image_vec, "text": text_vec},
            {
                "kind": "taught",
                "label": label,
                "transcript": transcript,
                "ts": ts or time.time(),
                "thumb": thumb,
            },
        )

    def remember_sighting(self, image_vec, label, ts=None, thumb=None):
        """A cadence write: something stable in view, image vector only."""
        return self._upsert(
            {"image": image_vec},
            {
                "kind": "seen",
                "label": label,
                "ts": ts or time.time(),
                "thumb": thumb,
            },
        )

    # -- reads ----------------------------------------------------------------

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
        heard = self.shard.query(QueryRequest(
            query=Query.Nearest(text_vec, using="text"),
            filter=window,
            limit=limit,
            with_payload=True,
        ))
        seen = self.shard.query(QueryRequest(
            query=Query.Nearest(clip_text_vec, using="image"),
            filter=window,
            limit=limit,
            with_payload=True,
        ))
        return {"seen": seen, "heard": heard}
