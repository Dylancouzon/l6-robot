# l6-robot

![The memory robot — a frosted pebble with a single camera eye and an LED field flickering like an HNSW graph traversal](assets/robot-concept.png)

> **Frozen course build.** A cut-down snapshot of [qdrant-labs/memory-fleet](https://github.com/qdrant-labs/memory-fleet), simplified for L6 of the Deep Learning course "Building On-Device AI Memory with Qdrant Edge." memory-fleet stays the living project; this repo is the version captured for the course and doesn't track its changes.

Instructor-side demo, never shipped to students. A memory loop running live on camera and mic: capture → detect → embed → match → teach.

Same stack as the course notebooks: one Qdrant Edge shard (0.7.2) with two named vectors — `text` 768 (Nomic v1.5) and `image` 512 (CLIP ViT-B/32), both via FastEmbed — Whisper-base via onnx-asr for speech, and the same `RECOGNIZE_THRESHOLD = 0.80` nearest-match check as L5. Detection is YOLOE prompt-free, and the labels it produces are discarded — memory decides what a thing is, not the detector. Weights auto-download via ultralytics, or copy `yoloe-11l-seg-pf.pt` into the repo root.

## Run

```bash
uv sync
uv run python -m robot.app                 # live on this machine -> http://127.0.0.1:8765
uv run python -m robot.app --host 0.0.0.0  # live, reachable from a phone/iPad on the network
uv run python -m robot.app --source d/     # headless: replay an image dir or video
uv run python smoke_test.py                # file-mode end-to-end check
```

The live view is a browser page (OpenCV's macOS windows misbehave on multi-monitor setups). Controls, in the tab or as touch buttons: **T** teach the focused object by voice ("this is my mug — Maria made it"), **A** ask a question by voice ("what did you see today?"), **R** reboot (close the shard, reload from disk, re-ask), **Q** quit. Every recognized object is shown; only the most prominent unknown is teachable. Calibration knobs: `--threshold` (recognition), `--conf` (detector sensitivity).

**Companion view (phone/iPad).** `--host 0.0.0.0` serves the page over HTTPS (a self-signed cert is generated into `cert/` on first run) and prints a `https://<lan-ip>:8765` URL. Open it on the phone, accept the certificate warning once, then **hold TEACH or ASK to record from the phone's own mic** — the clip uploads to the robot, which transcribes and writes the memory. The screen stays awake, and the view fills the phone held in landscape. Two ways to connect, both offline: share a Wi-Fi network, or run the robot's own hotspot — `sudo nmcli device wifi hotspot ssid l6-robot password <pw>`, then `https://10.42.0.1:8765`.

On a laptop the keyboard T/A keys record from the machine's own mic via sounddevice (the phone hold-to-talk above is the demo mic); set `robot/audio.py:MIC_DEVICE` if the default input is wrong (`uv run python -c "import sounddevice; print(sounddevice.query_devices())"`).

## Layout

- `robot/models.py` — the course's embedding + speech stack, kept aligned with the course repo's `.build/utils/`.
- `robot/memory.py` — the Edge shard: teach (one point, both named vectors), recognize (nearest taught view vs 0.80), time-filtered day recall grouped seen vs heard.
- `robot/detect.py` — YOLOE + tracking + the stability/cadence gate (from qdrant-labs/memory-fleet).
- `robot/core.py` — the loop itself; both front ends drive it.
- `robot/app.py` — live browser view (local or over the network) and headless replay.
- `testdata/` — fixture images (Wikimedia Commons, see `CREDITS.json`) and WAVs for the smoke test.

Shard data lives in `edge-data/` (gitignored). Delete it for a blank memory.

## Robot hardware — the memory robot

One self-contained demo unit: a ~140 mm frosted "pebble" (grapefruit-sized) that runs the full loop untethered and films well in a kitchen, a studio, or on a booth table. It's a single build — a demo/booth prop, not a persistent home device.

- **Compute inside the body.** The Jetson Orin Nano Super 8 GB (secured) lies flat in the lower half as the weighted base — heaviest part low for stability, hot parts away from the camera. Intake vents underneath, exhaust out the back; the shell is *not* sealed or it thermal-throttles.
- **One camera "eye"** at front-center behind a clear lens — a UVC USB webcam, plug-and-play on Jetson.
- **An LED field across the shell.** Addressable LEDs scattered over the translucent upper half, animated as a rolling HNSW-style graph traversal (a node lights, neighbors ripple, a "current" point hops on). Driven straight off the Jetson's SPI header (APA102/DotStar) — **no second board**. These are ambient decoration, not a live data readout, so they stay outside the honest-claim contract; optionally gate the animation on the real recognize loop so each query fires a hop.
- **No on-robot mic, speaker, or screen.** The companion phone/iPad is the whole interface: the browser UI serves over the robot's network, touch hold-to-talk buttons replace the T/A/R/Q keys, and the phone's mic records teach/ask audio (uploaded to the robot, which still does all transcription and memory work). Near-field phone mic beats a far-field array in a loud hall. The filmed "MEMORY WRITTEN" card lives on this view.

Full parts list and prices (~$362 full, ~$295 budget) and the Jetson software port checklist live in **[BOM.md](BOM.md)**.
