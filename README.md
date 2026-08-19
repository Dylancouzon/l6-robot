# l6-robot

![The L6 robot: a printed shell with a camera behind its visor and an antenna on top](assets/robot-render.gif)

A robot that **sees an object, learns its name from your voice, and remembers where it saw it**. Everything runs on the Jetson Orin Nano: no cloud, no API key, no LLM. Recognition and recall are vector search over an embedded Qdrant Edge shard.

This is the instructor demo for L6 of [Building On-Device AI Memory with Qdrant Edge](https://github.com/Dylancouzon/SC-Qdrant-C3), a DeepLearning.AI short course from Qdrant.

```text
camera / mic -> detect -> embed -> match -> teach -> recall
```

## What It Does

**Point it at something and hold TEACH.** Say "this is my chair". The robot crops the object from the frame, embeds the crop with CLIP, embeds your sentence with Nomic, and writes one memory with both vectors.

**It recognizes the object from then on.** The panel names what it is looking at, shows the remembered view beside the live one, and compares the score with the recognition bar.

<img src="assets/screens/recognize.png" alt="The phone view: a room with teal boxes reading dylan 0.91, chair 0.97, and keyboard 0.96, over a panel showing chair at 0.970 against a bar of 0.90" width="330">

**Everything it knows is browsable.** Each card holds the object's picture, the sentence you taught it with, and how many times it has been seen since. Rename fixes a misheard name without changing the vectors. Forget removes the object.

<img src="assets/screens/memory.png" alt="The memory tab: cards for dylan, keyboard, chair, and smartphone, each with a photo, its taught sentence, a sighting count, and Rename and Forget buttons" width="330">

**Ask where something is.** "When did you last see Dylan?" is matched against the sentence you taught it with. The robot answers with the last few times it saw the object, each with a time, a place, and the photo it took.

<img src="assets/screens/recall.png" alt="A recall answer reading I saw dylan at 3:17 AM, in Dylan's room, before that at 2:42 AM and 2:21 AM, above the photos the robot took" width="330">

**It can be taught from near misses.** An orange `hat? 0.87` box means the nearest taught object came close to the bar without clearing it. Tap the label to confirm it, and the robot learns that angle.

**Power is the off switch.** Every memory is flushed to disk as it is written. Press `R` to close the shard, reload it from disk, and answer again from what is genuinely on the device.

## What It Runs

| Piece | What |
|---|---|
| Vector search | Qdrant Edge 0.7.2, embedded, one shard on disk |
| Vectors | `image` 512-dim CLIP ViT-B/32, `text` 768-dim Nomic v1.5, both through FastEmbed |
| Speech | Whisper-base through `onnx-asr`, on CPU |
| Detection | YOLOE prompt-free, with BoT-SORT tracking |
| Recognition rule | nearest taught view must score at least `RECOGNIZE_THRESHOLD` |

The detector's own class labels are thrown away. Detection finds *a thing*. Memory decides *which* thing, so the robot can learn an object no detector has a word for.

## Run It

```bash
uv sync
cp .env.example .env
uv run python -m robot.app
```

That opens a browser view at `http://127.0.0.1:8765`. YOLO weights download on first run, along with about 1.5 GB of models.

The values in `.env` are tuned for one specific camera. If recognition looks wrong, read [Calibrating For Your Camera](#calibrating-for-your-camera).

## Controls

| Control | Action |
|---|---|
| `T` / hold **TEACH** | Teach the focused unknown object by voice |
| `A` / hold **ASK** | Ask a question by voice |
| `M` / **MEMORY** | Everything it knows, with pictures. Delete, rename, and un-ignore from here |
| `Q` / **IGNORE** | Dismiss the current unknown, and keep it dismissed across restarts |
| `F` | Forget the focused recognized object. Keyboard only |
| `R` | Close the shard and reload it from disk. Keyboard only |
| Ctrl-C | Quit. There is no on-screen quit, so a stray tap cannot end the demo |

**TEACH acts on the most prominent unknown object**, where prominent means large and near the center of the frame. To aim the robot, bring the object closer and hold it toward the middle. Focus is sticky: an object keeps the thick box until something is clearly more prominent, so a button press lands on the object you meant.

Delete from the memory tab, where you can pick an object from a list and see its picture first. Camera-aimed deletion is keyboard-only because focus can move between deciding to press and pressing.

## Teaching It Well

Two habits matter more than any setting.

**Teach each object two or three times, turning it between teaches.** One taught view is one point in CLIP's space, and the same object from another angle can land far from it. Re-teaching adds views. Recognition matches the nearest one. Measured on a live shard: a laptop's best single view covered 103 of its 173 later sightings, and the union of its six views covered all 173.

**Speak a phrase, not a single word.** "This is my laptop" works better than "laptop", because Whisper-base uses surrounding words to understand a short word. The same phrase is also what later questions match against.

Teach objects rather than surfaces. A crop with little in it, such as a blank panel or a reflective surface, can match too much of the room. Fill more of the frame with the object and teach it again.

People are objects too. Point the camera at someone, say "this is Dylan", and they are recognized like anything else. CLIP is not a face recognizer: it keys on the whole silhouette, clothing included, so expect to re-teach when someone changes their jacket.

## Calibrating For Your Camera

**Symptom:** you teach one object and half the room starts matching it, just over the bar.

CLIP cosine similarity, a measure of how close two vectors point, does not behave like a percent from 0 to 1. Two unrelated crops from the same camera can routinely score 0.75 to 0.85, because they share lighting, sensor, background, and scale. `0.90` is not "90% confident". It is a point above that floor. Move the camera farther away and every crop gets smaller and softer, the floor rises, and a threshold that worked at arm's length starts matching the furniture.

```bash
uv run python testdata/verify_scores.py
```

The script crops through the same code path the live robot uses and prints same-object and different-object score ranges, the margin between them, and a threshold sweep. Put a value from the clean range into `.env`. To calibrate against your own scene, photograph two or three objects three times each, name them `<object>_<n>.jpg`, and pass `--source ~/my-photos`.

Separate photos of distinct objects score farther apart than a live cluttered scene does, so treat the script's answer as a floor and expect to raise it against the real thing.

The other knobs in `.env` are the size band the detector keeps. `DETECT_MAX_AREA` drops boxes bigger than a fraction of the frame, which a prompt-free detector proposes freely for walls and desks. `DETECT_MIN_AREA` drops the small far-away clutter that keeps stealing the unknown box. Both are *areas*, so they move as the square of apparent size.

## Phone Or Tablet Demo

```bash
uv run python -m robot.app --host 0.0.0.0
```

The app prints an HTTPS URL. Open it on the phone, accept the certificate warning once, then hold the on-screen buttons to talk. The phone records the audio and uploads it. The robot does everything else.

Browsers only allow microphone access in a secure context, so the app serves HTTPS and generates a self-signed certificate on first run. Self-signed means the robot created the certificate itself instead of getting one from a public certificate authority, so the browser warns you. No public certificate authority will sign a certificate for a private address.

**To remove the warning on your demo phone**, install the certificate as trusted:

1. Open `https://<address>:8765/cert.crt` and accept the warning one last time. **On iOS this must be done in Safari**, which is the only browser that hands the file to the system as an installable profile.
2. **iOS:** Settings > General > VPN & Device Management, install the profile, then Settings > General > About > **Certificate Trust Settings** and enable full trust. That last step is easy to miss.
3. **Android:** Settings > Security > Encryption & credentials > Install a certificate > **CA certificate**.

On iOS the trust is system-wide, so every browser on the phone stops warning and microphone grants start being remembered. The certificate names the address it was generated for, so trust it once per address you demo on.

The page opens the microphone on your first touch and keeps it open, which is why the recording indicator stays on. Opening it per press takes long enough that the start of a promptly spoken word is lost.

## Headless Appliance

The demo unit has no keyboard, no screen, and no network it can rely on. As an appliance, it boots into the robot on **its own Wi-Fi network**.

```bash
sudo ./deploy/headless-setup.sh
sudo reboot
```

Then, from a phone anywhere, including a venue with no Wi-Fi at all:

1. Join the network **`l6-robot`**, password **`qdrantedge`**.
2. Open **`https://10.42.0.1:8765`**.
3. Accept the certificate once, or install it. This trust is permanent, because the hotspot address never changes.

Set your own name and password with `sudo SSID=my-robot PSK=my-password ./deploy/headless-setup.sh`.

**Your phone keeps its cellular service.** Joining only takes over Wi-Fi, and both iOS and Android keep internet traffic on mobile data once they notice the robot's network has no internet path.

**Do not leave a laptop joined to the hotspot.** When the robot has Ethernet, its hotspot shares that connection, so a laptop routes all its background traffic through the same radio that carries the video feed. That is the most common cause of a demo that looks slow.

The setup changes five things, each reversible on its own:

| Change | Why | Undo |
|---|---|---|
| Installs and enables `l6-robot.service` | Starts at boot, restarts on failure | `sudo systemctl disable --now l6-robot` |
| Adds an `l6-hotspot` profile on 5 GHz channel 44, and stops saved networks autoconnecting | One radio cannot be an access point and a client at once. 2.4 GHz cannot carry the video feed | `sudo nmcli con delete l6-hotspot` |
| `systemctl set-default multi-user.target` | Frees roughly 1.5 GB of desktop on an 8 GB board | `sudo systemctl set-default graphical.target` |
| Generates ssh host keys and starts `sshd` | The only way into a box with no peripherals | |
| Fills the model cache under `$HOME` | FastEmbed defaults to `/tmp`, which is pruned at 30 days. A robot with no internet would boot into a download that never finishes | |

Maintenance is `ssh qdrant@10.42.0.1`, then `journalctl -u l6-robot -f`. [REMOTE-ACCESS.md](REMOTE-ACCESS.md) covers the other ways in, the serial console fallback, and how to move the robot between Wi-Fi networks.

**Set the clock before filming.** Recall reads times out loud, and a robot that has been unplugged with no internet does not know what time it is. Either `sudo timedatectl set-time "..."` over ssh, or fit a coin cell to the RTC connector.

## Running On A Jetson Orin Nano

The same run commands work on the 8 GB board. Four things are worth knowing:

- **Torch prints a compute-capability warning.** It is cosmetic. Orin is `sm_87` and runs the wheel's `sm_80` kernels, and CUDA is genuinely in use: the detector runs at roughly 165 ms per frame against seconds per frame on CPU.
- **Only the CLIP vision encoder loads at startup.** Speech and text encoders load the first time you teach or ask, which is what keeps the board out of swap. Expect roughly 2.5 GB after startup.
- **Keep the stock 15 W power mode.** The bottleneck is memory and the USB camera, not the GPU.
- **"System throttled due to Over-current" can pop up repeatedly.** In this demo it does not mean the robot is too slow or underpowered. `OC3` counts instantaneous spikes caught by a hardware comparator, far too fast for the power sensor to sample, which is why `tegrastats` shows nothing near the limit. Sustained draw here is about 1.9 A against a 4.05 A limit, and detector latency stays flat. The appliance setup removes the desktop applet that shows the popup.

The live view is MJPEG, a stream of JPEG images sent as video. At 10 fps it is well over 10 Mbps over the robot's own hotspot, which is the narrowest part of the chain. If the feed lags, measure before changing settings:

```bash
iw dev wlP1p1s0 info    # which band and channel the access point is really on
timeout 5 curl -sk https://127.0.0.1:8765/stream -o /dev/null -w '%{size_download}\n'
```

The second command measures what the robot wants to send with the radio out of the picture. If that number is comfortable and the phone still lags, the problem is the link. `STREAM_QUALITY` in `robot/device/live.py` is the one number to turn, against a measurement of your own scene.

## Project Layout

The package is split by what you are trying to understand. **`device` imports `brain`, and `brain` never imports `device`**, so the retrieval code can be read without a camera or a socket in the way.

```text
robot/app.py            argparse, main, and headless replay
robot/config.py         per-camera knobs, read from .env
robot/brain/            the retrieval and the decision loop
  core.py               Robot: the loop, attention, teach / forget / ask
  memory.py             Qdrant Edge: the shard, the threshold, recall
  models.py             CLIP, Nomic, and Whisper loaders
  detect.py             YOLOE, tracking, and the stability gate
  labels.py             a spoken sentence to an object's name
robot/device/           camera, browser, microphone, hardware
  live.py               the threads, the buttons, and the voice path
  server.py             HTTP routes, the MJPEG stream, and the certificate
  page.html             the whole user interface
  mic.py                local microphone capture
  draw.py               boxes drawn onto the feed
deploy/                 the setup script and systemd unit for the appliance
testdata/               replay fixtures, and verify_scores.py for calibration
hardware/               the enclosure: print files, the model, and the spec
```

Memories live in `edge-data/`, which is gitignored. Delete that directory for a blank robot, or start with `--reset`.

Start reading at `robot/brain/core.py`. `process_frame` is the loop, and everything else is a step inside it.

To run the recognition path with no camera at all:

```bash
uv run python -m robot.app --source testdata/
```

## Flags

| Flag | What it does |
|---|---|
| `--threshold 0.90` | Recognition bar. `.env` `RECOGNIZE_THRESHOLD` |
| `--conf 0.30` | Detector confidence floor. Raise it to track less clutter. `.env` `DETECT_CONF` |
| `--max-area 0.20` | Biggest detection kept, as a fraction of the frame. `.env` `DETECT_MAX_AREA` |
| `--location "Hotel room"` | Place stamped on this session's memories, which recall reads back |
| `--reset` | Wipe all memories before starting |
| `--camera 1` | Use a different webcam |
| `--host 0.0.0.0` | Serve the view on the network, for a phone or iPad |
| `--advertise 10.42.0.1` | Address to put in the URL and the certificate, for the appliance |
| `--watchdog 30` | Exit if the camera delivers no frame for this long, so a service manager can restart. Off by default, because a closed laptop lid looks exactly like a stalled camera |
| `--source DIR` | Headless replay over images or a video, with no camera |

The first three are per-camera settings whose permanent home is `.env`. The flags override it for one run, which is useful while calibrating.

## Hardware

A 130 x 190 mm printed shell with a Jetson standing on edge inside it, one USB camera behind the eye, and an antenna on top. It is a filming and booth prop rather than a consumer device.

- **Jetson Orin Nano Super 8 GB**, low in the shell, where its weight keeps the robot upright.
- **A 37 x 37 mm USB (UVC) camera** as the eye. The printed tray is cut to that size, so measure yours before printing.
- **No microphone, speaker, or screen.** Your phone browser is the whole interface. Spoken answers use macOS `say` when you run this on a Mac; on the Jetson the same sentence appears on the panel.
- **Six printed parts, no screws.** A twist lock, two snap clips, one zip tie, and a drop of glue on the visor.

Parts and prices are in [BOM.md](BOM.md). Printing and assembly are in [hardware/README.md](hardware/README.md).

A Raspberry Pi 5 with a USB webcam should run the software, but CPU-only inference makes it slow and it does not fit this shell. It is not a tested target.
