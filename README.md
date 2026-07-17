# l6-robot

![Memory robot concept](assets/robot-concept.png)

`l6-robot` is the instructor demo for L6 of **Building On-Device AI Memory with Qdrant Edge**. It is a frozen, course-sized snapshot of [qdrant-labs/memory-fleet](https://github.com/qdrant-labs/memory-fleet), trimmed to show one clear loop:

```text
camera / mic -> detect -> embed -> match -> teach -> recall
```

No LLM runs in this loop. Recognition and recall are vector search and retrieval, not generation.

This repo is not the living product repo. It is the version used for the lesson.

## What It Runs

- **Vector search:** Qdrant Edge `0.7.2` (embedded, on-device)
- **Vectors:** `text` 768-dim Nomic v1.5 and `image` 512-dim CLIP ViT-B/32, both through FastEmbed
- **Speech:** Whisper-base through `onnx-asr`
- **Detection:** YOLOE prompt-free
- **Recognition rule:** nearest taught view must meet `RECOGNIZE_THRESHOLD = 0.80`

Detector labels are only used to crop objects. The memory layer decides what an object is.

YOLO weights download automatically through Ultralytics. You can also place `yoloe-11l-seg-pf.pt` in the repo root.

## Run

```bash
uv sync
uv run python -m robot.app
```

The app opens a browser view at `http://127.0.0.1:8765`.

Useful variants:

```bash
uv run python -m robot.app --host 0.0.0.0  # phone/iPad on the same network
uv run python -m robot.app --source d/     # replay an image directory or video
```

## Controls

| Control | Action |
|---|---|
| `T` / hold **TEACH** | Teach the focused unknown object by voice |
| `A` / hold **ASK** | Ask a voice question, such as "what did you see today?" |
| `R` / **REBOOT** | Close the shard, reload from disk, then re-ask |
| `F` / **FORGET** | Delete what it knows about the focused recognized object |
| `Q` / **IGNORE** | Dismiss the current unknown (clutter you won't teach) |
| Ctrl-C | Quit (no on-screen quit — a stray tap won't end the demo) |

Every recognized object is drawn on screen. Only the most prominent unknown object is teachable.

Two useful tuning flags:

```bash
uv run python -m robot.app --threshold 0.80
uv run python -m robot.app --conf 0.35
```

## Phone Or Tablet Demo

Run the app on the robot or laptop with:

```bash
uv run python -m robot.app --host 0.0.0.0
```

The app prints an HTTPS LAN URL such as `https://<lan-ip>:8765`. Open that URL on a phone or iPad, accept the self-signed certificate once, then use the on-screen hold-to-talk buttons. The phone records the audio and uploads it; the robot still handles transcription, memory writes, and recall.

This works offline in either setup:

- Put the robot and phone on the same Wi-Fi network.
- Or make the robot a hotspot:

```bash
sudo nmcli device wifi hotspot ssid l6-robot password <password>
```

Then open `https://10.42.0.1:8765`.

On a laptop, the `T` and `A` keys use the laptop mic through `sounddevice`. If the wrong input is selected, set `MIC_DEVICE` in `robot/audio.py`. To list devices:

```bash
uv run python -c "import sounddevice; print(sounddevice.query_devices())"
```

## Project Layout

| Path | Purpose |
|---|---|
| `robot/app.py` | Browser UI, phone controls, live mode, and replay mode |
| `robot/core.py` | Main robot loop |
| `robot/detect.py` | YOLOE detection, tracking, and cadence gating |
| `robot/memory.py` | Qdrant Edge teach, recognize, and day-recall logic |
| `robot/models.py` | Embedding and speech model setup |
| `testdata/` | Replay fixtures: images and WAVs |

Shard data is stored in `edge-data/`, which is gitignored. Delete that directory for a blank memory.

## Hardware

The intended hardware is a single demo unit: a frosted, grapefruit-sized shell with a Jetson inside, one USB camera, and a small addressable LED field. It is a filming and booth prop, not a consumer device.

High-level design:

- Jetson Orin Nano Super 8 GB inside the base
- UVC USB camera as the front "eye"
- APA102/DotStar LEDs driven from the Jetson SPI header
- No built-in mic, speaker, or screen; your phone browser is the interface

The full parts list, prices, build tiers, and Jetson port notes are in [BOM.md](BOM.md).

## Build Tiers

**Full build: Jetson Orin Nano Super 8 GB.** The default target. It has CUDA headroom for the detector and the rest of the memory stack.

A Raspberry Pi 5 (8/16 GB) + USB webcam is experimental only: it should run, but CPU-only inference makes it should and the detector would need manual tuning.

The build depends on detector-visible objects. 
