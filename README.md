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

## V2: an introduction to the fleet (planned, not built)

The fleet concept — memories shared across devices through the cloud — demonstrated with one robot and sequential sessions, no fleet management rebuilt:

1. **Kitchen session**: teach objects at home. At session end, the local shard's memories upload to a Qdrant Cloud collection (curated/reviewed there as needed).
2. **Reset**: the robot's local memory is wiped — on camera; the wipe is part of the story.
3. **Studio session**: at startup the robot pulls the collection into a fresh local shard. It has never seen the kitchen — but its memory has: "where are my keys?" answers with the kitchen sighting, photo and time, while new studio teaching keeps working and uploads the same way.

### Design rules

- **Append-only.** Devices/sessions push points and pull everyone's; nothing is merged, folded, or decayed. Same object taught twice just means two taught views — nearest-match already handles that. This is the line that keeps V2 from becoming a second [memory-fleet](https://github.com/qdrant-labs/memory-fleet).
- After the startup pull, everything stays local and offline, exactly as today.
- Out of scope: any LLM answer layer, auto-labeling from detector classes, live two-way sync between simultaneously-running units.

### Build plan

- **Prerequisite refactor — portable thumbnails**: `Robot._thumb` stops writing files; instead each point's payload carries `thumb_b64` (crop resized to ~96 px, JPEG, base64, ~2–3 KB). The answer card decodes with `cv2.imdecode`. Delete the `thumbs/` dir handling. A memory is then complete wherever it lands.
- **Cloud collection**: one collection (e.g. `robot_memories`) with the exact local shape — named vectors `text` 768 / `image` 512, cosine — created once with qdrant-client. Payload schema identical to local. Point ids are already unique across sessions (`time_ns`-seeded), so upserts are idempotent and append-only falls out for free.
- **`robot/sync.py`** with two entry points, runnable standalone:
  - `uv run python -m robot.sync push [--data edge-data]` — `ScrollRequest` over the local shard, batch upsert to Cloud via qdrant-client.
  - `uv run python -m robot.sync pull [--data edge-data]` — bring the collection down into a fresh local shard. Preferred path is Edge's built-in snapshot flow (`snapshot_manifest` / `unpack_snapshot` / `update_from_snapshot` — we own only the transport); `../hive-mind/fleetmemory/sync/` is the working reference for that transport code. Check the current qdrant-edge docs (skills.qdrant.tech) before writing it — the API is beta and moves.
- **App integration**: `--sync` flag on `robot.app` = pull at startup, push on quit. Sync failure must degrade loudly but safely: banner "cloud unreachable — running on local memory only", never a crash, never a half-pulled shard (pull into a temp dir, swap on success).
- **Config**: `.env` with `QDRANT_URL`, `QDRANT_API_KEY`, `DEVICE_NAME` (stamped into each point's payload), mirroring memory-fleet's config.py. No config beyond these three values.
- **Verification**: extend the smoke test with a sync leg — teach locally → push to a scratch collection → wipe local → pull → held-out recognition and the day question still pass. Needs a reachable cluster (free tier is fine); skip the leg with a warning when `QDRANT_URL` is unset so the offline smoke test stays green.
- **Demo choreography note**: the studio pull happens once, at startup, on camera; after it, airplane mode back on — the offline-reboot beat still holds because the session runs local.

## Robot hardware (scoping — parts not yet picked)

Target: one self-contained unit that runs the full loop untethered, filmable in a kitchen and a studio. The platform is fixed (Jetson Orin Nano per the course shotlist); peripherals below are requirements plus a default pick, not final choices.

| Part | Requirement | Default pick | Why / watch out |
|---|---|---|---|
| Compute | aarch64, ≥8 GB RAM, CUDA for YOLOE | **Jetson Orin Nano 8 GB dev kit** | The course names it ("hardware that costs about as much as a textbook stack"). 8 GB fits models (~2 GB peak) + torch comfortably. |
| Storage | ≥128 GB, fast enough for ~1.5 GB of model loads | **256 GB NVMe SSD** (M.2 2280) | Dev kit boots from microSD but model load times and torch installs are painful there; NVMe is the standard Jetson move. |
| Camera | UVC USB webcam, 720p+ | Logitech C920-class | USB/UVC works with OpenCV out of the box on Jetson. Avoid CSI ribbon cameras: driver + `cv2.VideoCapture` friction for zero demo benefit. |
| Mic | USB, works as default input | Small USB conference mic (or the fifine) | Same `sounddevice` path as the laptop. Needs `portaudio19-dev` via apt. |
| Power (bench) | 19 V DC barrel, ~25 W | The dev kit PSU | — |
| Power (untethered) | 7–20 V DC, ≥30 Wh for a ~1 h session | USB-C PD power bank + PD trigger cable (20 V) to barrel | This is what makes it a "robot" on camera instead of a desktop. Test sustained draw under YOLOE load before the shoot. |
| Display for the live view | Anything with a browser | A laptop/tablet on the same network | The app already serves the view as a web page; bind to `0.0.0.0` (small change) and the screen-capture machine just opens the URL. An on-robot 7" HDMI panel is optional polish, not required. |
| Case | Holds Jetson + battery + camera + mic, camera at object height | 3D-printed per the course shotlist | B-roll shot 8 shows the case open — leave the internals presentable. |

**Jetson software port checklist (verify before buying anything else):**

1. `qdrant-edge-py==0.7.2` ships an aarch64 Linux wheel — check first; it's the one dependency without an obvious fallback.
2. PyTorch + ultralytics from NVIDIA's JetPack wheels; device selection in `robot/detect.py` becomes `cuda` (currently `mps`/`cpu` — one line).
3. `onnxruntime` (CPU) aarch64 wheel for fastembed + onnx-asr — exists, but pin what you test.
4. `sounddevice` needs `sudo apt install portaudio19-dev`; the macOS `objc` autorelease import already no-ops off-macOS.
5. TensorRT export of YOLOE is an optimization, not a gate — CPU/CUDA inference at ~5 fps detect cadence is enough for the demo. Do it last, if at all.
