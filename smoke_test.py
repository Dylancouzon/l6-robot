"""End-to-end smoke test on files — no camera, no mic.

Teach two views + one WAV -> recognize a held-out view >= 0.80 -> foreign
object stays UNKNOWN -> day question returns the taught memory -> close,
reload from disk -> recognition and the day question both repeat.

    uv run python smoke_test.py
"""
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2

from robot import models
from robot.core import Robot
from robot.memory import RECOGNIZE_THRESHOLD

TESTDATA = Path("testdata")
DATA_DIR = "edge-data-test"


def crop_of(robot, image_path, now):
    """Drive the real pipeline: 3 frames to pass the stability gate."""
    frame = cv2.imread(str(image_path))
    assert frame is not None, f"unreadable image: {image_path}"
    robot.detector.reset()
    for i in range(3):
        tracks = robot.detector.process(frame, now + 0.1 * i)
    focused = robot.focused(tracks)
    assert focused is not None, f"no stable detection in {image_path}"
    return focused.crop


def main():
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    robot = Robot(data_dir=DATA_DIR)
    now = time.time()

    groups = defaultdict(list)
    for p in sorted(TESTDATA.glob("*.jpg")):
        groups[p.stem.rsplit("_", 1)[0]].append(p)
    subject = max(groups, key=lambda k: len(groups[k]))
    foreign = min(groups, key=lambda k: len(groups[k]))
    views = groups[subject]
    assert len(views) >= 3 and subject != foreign, f"bad fixtures: {dict(groups)}"
    teach_views, held_out = views[:2], views[-1]

    # teach: two crops, one spoken sentence
    taught = None
    for v in teach_views:
        taught = robot.teach(crop_of(robot, v, now), TESTDATA / "teach.wav")
    print(f'taught "{taught["label"]}": "{taught["transcript"]}"')

    def check_recognition():
        # the live loop itself: stability gate -> embed -> match -> sighting
        frame = cv2.imread(str(held_out))
        robot.detector.reset()
        t0 = time.time()
        for i in range(3):
            tracks = robot.process_frame(frame, t0 + 0.1 * i)
        f = robot.focused(tracks)
        assert f is not None and f.label == taught["label"], (
            f"held-out view NOT recognized: label={f and f.label} "
            f"score={f and f.score:.3f}")
        assert f.score >= RECOGNIZE_THRESHOLD
        assert f.note == taught["transcript"]
        print(f"recognized {held_out.name} as \"{f.label}\": "
              f"{f.score:.3f} >= {RECOGNIZE_THRESHOLD}")

        fvec = models.embed_crop(crop_of(robot, groups[foreign][0], time.time()))
        fhit, fscore = robot.memory.recognize(fvec)
        assert fhit is None, f"foreign object wrongly recognized: {fscore:.3f}"
        print(f"foreign {foreign} stays unknown: {fscore:.3f} < "
              f"{RECOGNIZE_THRESHOLD}")

    def check_day_question():
        q, res = robot.ask_from_wav(TESTDATA / "ask.wav")
        heard = [h.payload.get("transcript") for h in res["heard"]]
        assert taught["transcript"] in heard, (
            f"day question missed the taught memory. Q={q!r} heard={heard}")
        assert res["seen"], "day question returned nothing in the seen group"
        print(f'day question {q!r}: taught memory is in "heard" '
              f"(top score {res['heard'][0].score:.3f})")

    check_recognition()
    check_day_question()

    n = robot.reboot()
    print(f"reboot: shard reopened from disk with {n} memories")
    check_recognition()
    check_day_question()

    robot.close()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
