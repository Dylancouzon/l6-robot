# l6-robot

![Memory robot concept](assets/robot-concept.png)

`l6-robot` is the instructor demo for L6 of **Building On-Device AI Memory with Qdrant Edge**. It is a frozen, course-sized snapshot of [qdrant-labs/memory-fleet](https://github.com/qdrant-labs/memory-fleet), trimmed to show one clear loop:

```text
camera / mic -> detect -> embed -> match -> teach -> recall
```

No LLM runs in this loop. Recognition and recall are vector search and retrieval, not generation.

This repo is not the living product repo. It is the version used for the lesson.

## What You Already Built

Every stage of this loop comes from a lesson notebook. Two of them are new.

| Loop stage | Where it comes from |
|---|---|
| Mic to transcript | L4 |
| Frame, embed, match against the threshold | L5 |
| Store, recall, forget | L2 |
| Cross-modal recall over photos, voice, and text | L3 and L4 |
| One shard, two skills, offline | L5 |
| Detect and crop objects in a cluttered frame | New here |
| Deciding when a memory is worth writing | New here |

The two new stages stay black boxes on purpose. A detector finds *a thing*; the memory layer decides *which* thing.

## What It Runs

- **Vector search:** Qdrant Edge `0.7.2` (embedded, on-device)
- **Vectors:** `text` 768-dim Nomic v1.5 and `image` 512-dim CLIP ViT-B/32, both through FastEmbed
- **Speech:** Whisper-base through `onnx-asr`
- **Detection:** YOLOE prompt-free
- **Recognition rule:** nearest taught view must meet `RECOGNIZE_THRESHOLD` (default `0.90`, per-camera — see [Calibrating For Your Camera](#calibrating-for-your-camera))

Detector labels are only used to crop objects. The memory layer decides what an object is.

YOLO weights download automatically through Ultralytics. You can also place `yoloe-11l-seg-pf.pt` in the repo root.

## Run

```bash
uv sync
cp .env.example .env
uv run python -m robot.app
```

The app opens a browser view at `http://127.0.0.1:8765`. To open the view from a phone or iPad instead, see [Phone Or Tablet Demo](#phone-or-tablet-demo).

The defaults in `.env` are tuned for one specific camera. Read [Calibrating For Your Camera](#calibrating-for-your-camera) before you conclude that recognition is broken — the single most common symptom, everything in the room matching the last thing you taught, is a threshold that hasn't been calibrated.

### Running On A Jetson Orin Nano

The same two commands work. Four things are worth knowing on the 8 GB board:

- **Torch prints a compute-capability warning** ("No published PyTorch CUDA builds ... support this GPU"). It is harmless: Orin is `sm_87` and executes the wheel's `sm_80` kernels. CUDA is genuinely in use — the detector runs at roughly 165 ms per frame at `imgsz=640`, against seconds per frame on CPU. Confirm with:

  ```bash
  uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```

- **Only the CLIP vision encoder loads at startup** (see `warm_up` in `robot/models.py`). Speech and text encoders load the first time you teach or ask. Loading all four up front needs about 3.3 GB before the camera even opens, which pushes the board into swap — and a swapping robot updates the view once every several seconds. Expect roughly 2.5 GB after startup and 4 GB once you have taught something.

- **Keep the stock 15 W power mode.** The bottleneck here is memory and the USB camera, not the GPU, so `jetson_clocks` and MAXN buy nothing.

- **"System throttled due to Over-current" will pop up repeatedly, and it is expected.** `OC3` counts *instantaneous* power spikes on `VDD_IN`, caught by a hardware comparator far too fast for the INA3221 sensor to sample — which is why `tegrastats` shows you nowhere near the limit while the counter climbs. Bursty GPU inference trips it; sustained draw here is about 1.9 A against a 4.05 A critical limit, CPU clocks hold at maximum, and per-frame detector latency stays flat. `OC1` and `OC2` (the averaged thresholds) stay at zero. NVIDIA describes the throttle as protective, not damage. Check the counters with:

  ```bash
  cat /sys/devices/platform/soctherm-oc-event/hwmon/hwmon*/oc?_event_cnt
  ```

  The popup comes from the nvpmodel tray applet, which polls those counters every second and re-fires while any of them increments. To stop the notifications without touching the hardware protection, either tick **Disable notification** in the tray menu (resets at login) or hide the applet's autostart entry:

  ```bash
  cp /etc/xdg/autostart/nvpmodel_indicator.desktop ~/.config/autostart/
  echo "Hidden=true" >> ~/.config/autostart/nvpmodel_indicator.desktop
  ```

## Controls

| Control | Action |
|---|---|
| `T` / hold **TEACH** | Teach the focused unknown object by voice |
| `A` / hold **ASK** | Ask a voice question, such as "what did you see today?" |
| `R` / **REBOOT** | Close the shard, reload from disk, then re-ask |
| `F` / **FORGET** | Delete what it knows about the focused recognized object |
| `Q` / **IGNORE** | Dismiss the current unknown (clutter you won't teach) |
| Ctrl-C | Quit (no on-screen quit, so a stray tap won't end the demo) |

Every recognized object is drawn on screen. Only the most prominent unknown object is teachable.

### Aiming It

`TEACH` acts on the most prominent **unknown** object; `FORGET` acts on the recognized object the panel is showing. Prominence is size weighted by centrality, so the way to aim the robot is to bring the object closer and hold it toward the middle of the frame.

Focus is deliberately **sticky**: once an object holds the thick box, it keeps it until something is clearly more prominent — roughly 25% more — or until it leaves the frame. Without that, two objects of similar size and position trade the box several times a second, and you end up teaching whichever one happened to win on the frame you pressed. If focus won't leave an object, move the one you want closer or more central, or press `Q` / **IGNORE** to dismiss the current unknown outright.

To trade a twitchier box for an easier-to-move one, `FOCUS_MARGIN` in `robot/core.py` is the single number: lower is more responsive and more jittery, higher is calmer and more stubborn.

### Flags

The first three are per-camera settings whose permanent home is `.env`; the flags override it for a single run, which is what you want while calibrating.

| Flag | What it does |
|---|---|
| `--threshold 0.90` | Recognition bar: the nearest taught view must score at least this to count as a match. `.env` `RECOGNIZE_THRESHOLD`. See [Calibrating For Your Camera](#calibrating-for-your-camera). |
| `--conf 0.30` | Detector confidence floor. Raise it if the view tracks too much clutter. `.env` `DETECT_CONF`. |
| `--max-area 0.20` | Biggest detection kept, as a fraction of the frame. The default drops torso-sized boxes. `.env` `DETECT_MAX_AREA`. |
| `--location "Hotel room"` | Place stamped on every memory this session; recall says it back ("I saw my keys at 2:14 PM, in Hotel room"). |
| `--reset` | Wipe all memories before starting, for a clean slate between takes. Kept off the live UI so a stray tap can't erase the demo. |
| `--camera 1` | Use a different webcam. |
| `--host 0.0.0.0` | Serve the browser view on the network so a phone or iPad can open it. The app still runs on this machine. |

## Calibrating For Your Camera

**Symptom:** you teach the robot one object and half the room starts matching it, at scores just over the bar, while the real object sits higher. Nothing is broken. The threshold is a per-camera number and the default is not yours.

CLIP cosine similarity does not run from 0 to 1 in practice. Two *unrelated* crops from the same camera routinely score 0.75 to 0.85, because they share lighting, sensor, background, and scale. `0.90` is not "90% confident" — it is a point above that floor. Move the camera further from the desk and every crop gets smaller and softer, the floor rises, and a threshold that worked at arm's length starts matching the furniture.

Copy the example file and calibrate:

```bash
cp .env.example .env
uv run python testdata/verify_scores.py
```

The script crops through the same code path the live robot uses and prints three things:

- **Per-pair scores**, split into same-object and different-object.
- **The margin**, worst same-object score minus best different-object score. If this is negative, no threshold works and the crops are the problem — get closer, add light, fill more of the frame.
- **A threshold sweep**, showing how many true matches survive and how many false ones creep in at each candidate.

Put the middle of the clean range into `.env` as `RECOGNIZE_THRESHOLD`.

To calibrate against your own scene rather than the bundled photos, take three photos of each of two or three objects with the camera you'll demo with, name them `<object>_<n>.jpg`, and point the script at them:

```bash
uv run python testdata/verify_scores.py --source ~/my-photos
```

Photos of distinct objects score lower than a live cluttered scene, so treat the script's answer as a floor and expect to raise it a little against the real thing.

### What to teach

Teach objects, not surfaces. A crop with little information in it — a blank panel, a reflective surface, a blurred fragment — sits near the middle of CLIP's space, close to everything, so it makes a memory that matches most of the room. That looks exactly like a broken threshold and isn't one. Fill more of the frame with the object and teach it again.

Teach it while it is still in view. A box outlives its object by a fraction of a second, so the detector can ride out a dropped frame instead of blinking — press `TEACH` just after pulling the object away and you can capture the last crop of empty space, which is the blank-crop problem above. If a memory starts matching everything, `F` / **FORGET** it and teach again.

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
| `robot/config.py` | Per-camera settings, read from `.env` |
| `testdata/` | Replay fixtures, plus `verify_scores.py` for calibration |

Shard data is stored in `edge-data/`, which is gitignored. Delete that directory for a blank memory.

## Hardware

The intended hardware is a single demo unit: a frosted, grapefruit-sized shell with a Jetson inside, one USB camera, and a small addressable LED field. It is a filming and booth prop, not a consumer device.

High-level design:

- Jetson Orin Nano Super 8 GB inside the base
- UVC USB camera as the front "eye"
- APA102/DotStar LEDs driven from the Jetson SPI header
- No built-in mic, speaker, or screen; your phone browser is the interface
- Spoken answers use macOS `say`; on Jetson, swap in `espeak` plus a small USB speaker (without one, the answer stays on-screen only)

The full parts list, prices, build tiers, and Jetson port notes are in [BOM.md](BOM.md).

## Build Tiers

**Full build: Jetson Orin Nano Super 8 GB.** The default target. It has CUDA headroom for the detector and the rest of the memory stack.

A Raspberry Pi 5 (8/16 GB) + USB webcam is experimental only: it should run, but CPU-only inference makes it slow and the detector would need manual tuning.

The build depends on detector-visible objects.
