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

The app opens a browser view at `http://127.0.0.1:8765`. To open the view from a phone or iPad instead, see [Phone Or Tablet Demo](#phone-or-tablet-demo). For the demo unit in its case — no keyboard, no screen, its own Wi-Fi, starts on power — see [Headless Appliance](#headless-appliance).

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
| `--advertise 10.42.0.1` | Address to put in the URL and the certificate, when it is not the one to look up. Used by the headless appliance, whose own hotspot address never changes — see [Headless Appliance](#headless-appliance). |
| `--watchdog 30` | Exit if the camera delivers no frame for this long, so a service manager can restart the robot. Off by default: on a laptop, closing the lid looks exactly like a stalled camera. |

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

### What to say

**Speak a phrase, not a single word.** "This is my laptop" is recognized reliably; "laptop" on its own often is not. Whisper-base leans on surrounding words to pin down a short one, and a bare noun gives it nothing to lean on — a real capture of one isolated word came back as `La-caw`. The phrase is also what `parse_label` is built to read: it takes the words after "this is" and stops at the first new clause, so *"This is my mug, Maria made it"* teaches `my mug`.

**Release the button when you stop talking.** Holding it open for a few extra seconds used to cost twice: Whisper charges for every second of audio, and given a stretch of silence it starts repeating itself — a 7.7 s hold around 1.4 s of speech transcribed to `L L L L L L…` and took 16.3 s. Silence is now trimmed off both ends before transcription (2.0 s and 5.2 s for that same clip), so a long hold costs you little — **as long as the robot can find your speech in it**. Speak up: the trim looks for audio above a fixed bar, and a hold too quiet to clear it is passed to Whisper whole, repeated letters and all. The `rms` in the log is the number to watch.

The log prints what it heard and what it kept, which is where to look first when a label comes out wrong:

```
recorded level (rms): 1641
trimmed the hold down to 2.0 s of speech
taught "my laptop": 'This is my laptop.'
```

An `rms` of 0 means no audio reached the robot at all — that is a browser mic permission, not a recognition problem.

## Phone Or Tablet Demo

Run the app on the robot or laptop with:

```bash
uv run python -m robot.app --host 0.0.0.0
```

The app prints an HTTPS LAN URL such as `https://<lan-ip>:8765`. Open that URL on a phone or iPad, accept the certificate warning once, then use the on-screen hold-to-talk buttons. The phone records the audio and uploads it; the robot still handles transcription, memory writes, and recall.

The page opens the mic on your first touch anywhere and **keeps it open** while the tab lives, so your browser shows a recording indicator the whole time. That is deliberate: opening the device and compiling the audio worklet per press takes long enough that the start of a promptly-spoken word lands before recording begins, and it does that work on the same main thread that is painting the video feed. Close the tab to release the mic. Audio is captured at 16 kHz — what Whisper wants — so what goes up the link is a third of the bytes a 48 kHz capture would send.

### Why HTTPS, And Why The Warning

Browsers only hand out the microphone in a **secure context**. `http://localhost` counts as one, but `http://192.168.x.x` does not — so the moment the interface moves to your phone, the app has to serve HTTPS. It generates a self-signed certificate into `cert/` on first run.

The warning appears because nothing vouches for that certificate. It is not a misconfiguration and it cannot be coded away: a public certificate authority will not sign a certificate for a private LAN address, and reaching one would need internet access this demo is designed not to need. Accepting it once per device is the normal path, and the mic works fine afterwards.

**To get rid of the warning on your own demo phone** — worth doing before filming — install the certificate as trusted, once:

1. Open `https://<lan-ip>:8765/cert.crt` on the phone and accept the warning one last time to download it.
2. **iOS/iPadOS:** Settings → General → VPN & Device Management → install the downloaded profile. Then, and this step is easy to miss, Settings → General → About → **Certificate Trust Settings** → enable full trust for `l6-robot`.
3. **Android:** Settings → Security → Encryption & credentials → Install a certificate → **CA certificate** → pick the downloaded file.

Reload the page and the warning is gone for good on that device.

The certificate names the IP address it was generated for, so it is regenerated automatically whenever the robot's address changes — which means a phone you trusted on your home Wi-Fi will warn again on the booth network, or after switching to hotspot mode. Trust it once per address you demo on. Certificates are valid for 397 days; Safari rejects anything much longer, trusted or not.

This works offline in either setup:

- Put the robot and phone on the same Wi-Fi network.
- Or make the robot a hotspot:

```bash
sudo nmcli device wifi hotspot ssid l6-robot password <password> band a channel 44
uv run python -m robot.app --host 0.0.0.0 --advertise 10.42.0.1
```

Then open `https://10.42.0.1:8765`. `--advertise` is what makes the certificate name the hotspot's address: without it the app names whichever address it can look up — the Ethernet one, or `127.0.0.1` when the hotspot is the only network — and a certificate that names an address you are not opening gives the harsher warning described above, the one trusting it cannot fix.

For the permanent version of this, where the robot boots into the hotspot by itself, see [Headless Appliance](#headless-appliance).

On a laptop, the `T` and `A` keys use the laptop mic through `sounddevice`. If the wrong input is selected, set `MIC_DEVICE` in `robot/audio.py`. To list devices:

```bash
uv run python -c "import sounddevice; print(sounddevice.query_devices())"
```

## Headless Appliance

The demo unit has no keyboard, no screen, and no network it can rely on. Set up as an appliance, it needs none of them: apply power and it boots into the robot, brings up **its own Wi-Fi network**, and stays live until the power goes away.

```bash
sudo ./deploy/headless-setup.sh
sudo reboot
```

Then, from a phone — anywhere, including a booth with no Wi-Fi at all:

1. Join the Wi-Fi network **`l6-robot`**, password **`qdrantedge`**.
2. Open **`https://10.42.0.1:8765`**.
3. Accept the certificate once, or install it from `https://10.42.0.1:8765/cert.crt` to stop being asked. Unlike the LAN case, this trust is **permanent**: the robot's hotspot address never changes, so the certificate never has to be reissued.

Set your own network name and password by running the script with them: `sudo SSID=my-robot PSK=my-password ./deploy/headless-setup.sh`.

**Your phone keeps its cellular service.** Joining the robot only takes over the phone's Wi-Fi; both iOS and Android keep sending internet traffic over mobile data once they notice the robot's network has no internet path. iOS says "No Internet Connection" under the network name and works fine. On Android, *Settings → Network & internet → Internet → "Switch to mobile data automatically"* is the one setting that can drop the Wi-Fi association outright; turn it off for that phone if it misbehaves. And when the robot happens to have Ethernet plugged in, its hotspot shares that connection, so there is real internet on it too.

### Turning It On And Off

There is no power button, and the Jetson does not need one: *"By default, Jetson Orin Nano Developer Kit turns on automatically as soon as the included DC power supply is connected to the DC power jack"* ([user guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/howto.html)). Plugging the case in is the on switch.

**Pulling the plug is the off switch, and it is safe for the memories.** Every teach writes one point and flushes it to disk immediately — `Memory._upsert` does this on purpose, because the lesson's offline-reboot beat power-cycles the device. There is no buffered state to lose. For a graceful shutdown anyway, `ssh qdrant@10.42.0.1 sudo poweroff`.

If you would rather have a real button in the case, the carrier board has a button header for one — a short press then gives a clean soft shutdown as well as power-on. The pin assignments are in the *Jetson Orin Nano Developer Kit Carrier Board Specification* (Jetson Download Center); confirm them against your board rather than against a blog post, because the same header also carries the force-recovery and reset pins.

### The Clock, Which Recall Reads Out Loud

Worth knowing before it embarrasses you on camera: recall says *"I saw my keys at 2:14 PM"*, and a robot that has been unplugged and has no internet does not know what time it is. There is no NTP server on its own hotspot, and `systemd-timesyncd` restores the **last time it saw** at boot rather than the real one — so a robot switched on cold at a booth stamps new memories with the time it was last packed away.

Two fixes, either is enough:

- **Set it once over ssh** when you set up for the day: `sudo timedatectl set-ntp false && sudo timedatectl set-time "2026-08-12 09:30:00"`, then `sudo systemctl restart l6-robot`. Turn NTP back on (`set-ntp true`) when the robot is next on a real network. The restart is because track ageing still reads the wall clock, so a large step leaves the boxes on screen confused for a few seconds; the watchdog itself is immune to it.
- **Fit a coin cell to the RTC backup battery connector** on the carrier board, which keeps the clock running with the power off. It is a 2-pin 1.25 mm connector, `J3` in the [Carrier Board Specification](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_specification_sp.pdf) — confirm against your own board, since `J13` next to it is the fan. Discussion and a working socket part are in [this NVIDIA forum thread](https://forums.developer.nvidia.com/t/rtc-battery-on-jetson-orin-nano-developer-kit/296732).

Plugging Ethernet in for a minute also fixes it, whenever that is an option.

### What The Setup Changes

Five things, each reversible on its own:

| Change | Why | Undo |
|---|---|---|
| Installs and enables `l6-robot.service` | Starts the robot at boot and restarts it if it dies | `sudo systemctl disable --now l6-robot` |
| Adds an `l6-hotspot` Wi-Fi profile in AP mode on **5 GHz channel 44**, and sets saved networks to not autoconnect | One radio cannot be an access point and a client at once, and the robot must work where there is no network. The band is not incidental — see [The Feed Is The Bottleneck](#the-feed-is-the-bottleneck) | `sudo nmcli con delete l6-hotspot`, then re-enable autoconnect on your own network |
| `systemctl set-default multi-user.target` | Frees roughly 1.5 GB of GNOME on an 8 GB board, next to a robot process that reaches 3.4 GB. It also retires the over-current popup, which is a desktop applet | `sudo systemctl set-default graphical.target` |
| Generates ssh host keys and starts `sshd` | The only way into a box with no peripherals. A missing host key leaves `sshd` enabled but dead, which you discover at the worst moment | — |
| Fills the model cache under `$HOME` | `FastEmbed` defaults to `/tmp`, which `systemd-tmpfiles` prunes at 30 days. Losing 1.1 GB of encoders on a robot with no internet means it boots into a download that never finishes | — |

### Maintenance

From a laptop joined to `l6-robot`, or over Ethernet when the robot is on a desk — the view is served on every interface, though opening it on the Ethernet address warns about the certificate, which names the hotspot:

```bash
ssh qdrant@10.42.0.1
journalctl -u l6-robot -f          # the robot's console output
sudo systemctl restart l6-robot    # after editing code or .env
```

There are three independent ways in — your LAN with the robot wired to the router, the robot's own hotspot, and a USB-C cable that needs no network at all — plus a serial console for when none of them answer. [REMOTE-ACCESS.md](REMOTE-ACCESS.md) covers all of them, how to work on the robot once you are in, and how to move it between Wi-Fi networks without stranding it.

To get the robot back onto a real network for updates, plug in Ethernet, or `sudo nmcli con up "<your network>"` — the saved profiles are kept, just stopped from autoconnecting. Bring the hotspot back with `sudo nmcli con up l6-hotspot`.

Two behaviours worth knowing before you debug them:

- **A crash is invisible and self-healing.** The service restarts 10 seconds later and takes about 40 seconds to reload the detector, so an unplugged camera looks like a robot that is simply slow to come back. `journalctl -u l6-robot` is where the reason is.
- **The service runs with `--watchdog 30`.** A USB camera that wedges *inside* a driver call cannot be noticed by the thread stuck in it, so the app exits if no frame arrives for 30 seconds and lets systemd restart it. If you ever see a restart with no error above it, that was this.

### The Feed Is The Bottleneck

The live view is MJPEG, and the composed 1760x720 frame is about 183 KB. At the camera's 10 fps that is **~13 Mbps** of video the robot has to push over its own hotspot — far more than anything else the demo does.

That number is why the hotspot is on 5 GHz. A 2.4 GHz access point at 20 MHz carries 15–25 Mbps in an empty room and much less in a hall full of phones, so the radio becomes the narrowest part of the chain. When that happens the failure is not a dropped frame: TCP queues what it cannot send, so the feed arrives smooth but **seconds behind the room**, and the delay grows the longer you watch. Pressing TEACH on an object you can no longer see is the symptom that ruins a demo.

The trade is range. 5 GHz carries less far and through less material, so keep the phone in the same room as the robot rather than across a hall.

If the feed still lags — an unavoidably crowded band, or a client stuck on 2.4 GHz — the one number to turn is `STREAM_QUALITY` in `robot/app.py`. Dropping it from 85 to 70 takes the frame to ~134 KB and the stream to ~11 Mbps, for a slightly softer image. Check what you are actually up against first:

```bash
iw dev wlP1p1s0 info                      # which band and channel the AP is really on
timeout 5 curl -sk https://127.0.0.1:8765/stream -o /dev/null -w '%{size_download}\n'
```

The second command measures what the robot *wants* to send, over loopback, with the radio out of the picture. Divide by 5 for bytes per second. If that number is comfortable and the phone still lags, the problem is the link, not the robot.

## Project Layout

| Path | Purpose |
|---|---|
| `robot/app.py` | Browser UI, phone controls, live mode, and replay mode |
| `robot/core.py` | Main robot loop |
| `robot/detect.py` | YOLOE detection, tracking, and cadence gating |
| `robot/memory.py` | Qdrant Edge teach, recognize, and day-recall logic |
| `robot/models.py` | Embedding and speech model setup |
| `robot/config.py` | Per-camera settings, read from `.env` |
| `deploy/` | `headless-setup.sh` and the systemd unit that make it an appliance |
| `REMOTE-ACCESS.md` | Getting a shell on the headless unit, and moving it between Wi-Fi networks |
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
