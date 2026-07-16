# l6-robot

Instructor-side demo app for L6 of "Building On-Device AI Memory with Qdrant Edge". Never shipped to students. A continuously-running memory loop on a camera and mic: capture → detect → embed → match → teach.

Same stack as the course notebooks: one Qdrant Edge shard (0.7.2) with two named vectors — `text` 768 (Nomic v1.5) and `image` 512 (CLIP ViT-B/32), both via FastEmbed — Whisper-base via onnx-asr for speech, and the same `RECOGNIZE_THRESHOLD = 0.80` nearest-match check as L5. Detection is YOLOE prompt-free (labels discarded); the weights auto-download via ultralytics or copy `yoloe-11l-seg-pf.pt` into the repo root.

## Run

```bash
uv sync
uv run python -m robot.app              # live: webcam + mic -> browser view
uv run python -m robot.app --source d/  # headless: replay an image dir or video
uv run python smoke_test.py             # file-mode end-to-end check
```

The live view opens at http://127.0.0.1:8765 (a browser page, because OpenCV's macOS windows misbehave on multi-monitor setups). Keys, pressed in the browser tab: **T** teach the focused object by voice ("this is my mug — Maria made it"), **A** ask a question by voice ("what did you see today?"), **R** the reboot beat (close the shard, reload from disk, re-ask), **Q** quit. Every recognized object is shown; only the most prominent unknown is shown and teachable. Calibration knobs: `--threshold` (recognition), `--conf` (detector sensitivity).

Mic capture shells out to ffmpeg (`brew install ffmpeg`); set `robot/audio.py:MIC_DEVICE` if the default input isn't your mic.

## Layout

- `robot/models.py` — the course's embedding + speech stack, kept aligned with the course repo's `.build/utils/`.
- `robot/memory.py` — the Edge shard: teach (one point, both named vectors), recognize (nearest taught view vs 0.80), time-filtered day recall grouped seen vs heard.
- `robot/detect.py` — YOLOE + tracking + the stability/cadence gate (from qdrant-labs/memory-fleet).
- `robot/core.py` — the loop itself; both front ends drive it.
- `robot/app.py` — live window and headless replay.
- `testdata/` — fixture images (Wikimedia Commons, see `CREDITS.json`) and WAVs for the smoke test.

Shard data lives in `edge-data/` (gitignored). Delete it for a blank memory.
