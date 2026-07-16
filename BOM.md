# Bill of materials — the memory robot

One self-contained demo unit: a ~140 mm frosted "pebble" with the Jetson inside as a
weighted, vented base, a single camera "eye," and a field of addressable LEDs scattered
across the shell that flicker like an HNSW graph being walked. Single build — this is a
demo/booth prop, not a persistent home device, so there's no fleet-of-units line here.

Everything is driven by the Jetson directly. **No second microcontroller** — the LEDs use
SPI (APA102/DotStar), not the timing-critical one-wire WS2812 protocol, so the Jetson's own
40-pin header runs them without help.

![Concept render](assets/robot-concept.png)

## Parts

| # | Part | Default pick | ~USD | Notes / watch out |
|---|------|--------------|-----:|-------------------|
| 1 | Compute | **Jetson Orin Nano Super 8 GB dev kit** | 249 | Secured. aarch64 + CUDA for YOLOE; 8 GB fits the stack (~2 GB peak) with headroom. Ships with its 19 V barrel-jack PSU. |
| 2 | Storage | 256 GB NVMe SSD (M.2 2280) | 30 | Boot + model loads from microSD are painfully slow; NVMe is the standard Jetson move. |
| 3 | Camera ("eye") | Logitech C920-class USB (UVC) | 50 | Plug-and-play with `cv2.VideoCapture` on Jetson. Avoid CSI ribbon cams — driver friction, no demo benefit. A compact USB module (e.g. Arducam UVC) reads cleaner as the lens if you want to shrink the eye. |
| 4 | Companion device (display + mic + speaker) | Dylan's phone / iPad on a stand | 0 | The robot is headless and **carries no mic or speaker** — the phone is the whole interface. The browser view serves over the robot's network with touch hold-to-talk buttons; the phone's mic records and uploads to a `/audio` endpoint (requires self-signed **HTTPS** — iOS blocks `getUserMedia` on plain HTTP; see checklist). Near-field phone mic beats any far-field array in a loud hall, and push-to-talk sidesteps VAD-in-noise. The filmed "MEMORY WRITTEN" card lives here too, so the companion view is contract-required. |
| 6 | LED field | Adafruit DotStar / APA102 strand (~30–60 px) | 12 | The graph-traversal shimmer. SPI (clock + data) off the Jetson header — reliable, no timing games. **Power budget**: fed from the header's 5 V rail, so brightness must stay capped (~25% / ambient) — 60 px at full white pulls 3.5 A and will brown out the rail. If a brighter look is ever wanted, add a dedicated 5 V buck feed, not more header current. |
| 7 | Level shifter (optional) | 74AHCT125 | 2 | Only if the strand misbehaves at 3.3 V logic; shifts SPI to 5 V. |
| 8 | Untethered power | **65 W-class** USB-C PD power bank + PD-trigger (20 V) → barrel cable | 55 | What makes it a "robot" on camera vs. a desktop. 20 V is in spec — the dev kit's DC jack accepts 7–20 V. Size the bank at 65 W: MAXN is 25 W module-only, plus camera, NVMe and LEDs. Must sustain 20 V output without sagging or auto-sleeping at partial load; test sustained draw under YOLOE load before any shoot. |
| 9 | Enclosure | 3D-printed shell, matte PLA/PETG | 6 | Frosted/translucent upper for the LEDs, opaque vented base for the Jetson + fan. Matte finish — glossy throws glare on camera. |
| 10 | Fasteners + wiring | M2/M3 heat-set inserts, screws, JST leads, short USB cables | 15 | — |
| | | **Total** | **~$420** | Well within budget. Moving mic + speaker + display onto the companion phone cut ~$95 and two USB devices from the shell. |

## Headroom (the "plan for larger models" ask)

The 8 GB Orin Nano runs the current stack (YOLOE-11L-seg + CLIP ViT-B/32 + Nomic v1.5 +
whisper-base) comfortably. If a future version needs a bigger CLIP (ViT-L), a larger text
embedder, or a small local VLM running concurrently, the drop-in upgrade is a **Jetson Orin
NX 16 GB** — same JetPack image, so zero code changes, roughly 2× RAM and TOPS. Start on the
Nano Super; swap the module only if the workload outgrows 8 GB.

## How the LED field is driven

The LEDs are **ambient decoration** — a scattered node/edge pattern animated as a rolling
graph traversal (a node lights, its neighbors ripple, a "current" point hops on, trails fade).
They are *not* a live readout of real search results, so they sit outside the honest-claim
contract's "every score on screen is a live value" rule (see `CLAUDE.md`). If you want them
honest-by-construction rather than purely generative, gate the animation on the real recognize
loop — every query fires a hop — which is already happening ~continuously and needs no data
mapping.

Physically: LEDs scattered across the upper (translucent) half of the shell; the camera eye at
front-center; the frosted PLA diffuses each point into a soft orb.

## Software port checklist (verify before buying anything else)

1. `qdrant-edge-py==0.7.2` aarch64 Linux wheel — **verified on PyPI** (`manylinux_2_28_aarch64`,
   download-tested; JetPack 6 / Ubuntu 22.04 satisfies it). The one no-fallback dependency is covered.
2. **Flashing needs an x86 Ubuntu host** with SDK Manager to put JetPack 6 on the NVMe
   (or bootstrap via microSD, then move to NVMe). Zero-cost, but plan for it — it's the first step.
3. PyTorch + ultralytics from **NVIDIA's JetPack wheels**, not stock pip. Note the stock `uv.lock`
   resolves torch with sbsa/x86 CUDA deps on aarch64 that don't target Jetson's iGPU — `uv sync`
   as-is won't give a working CUDA torch. Ultralytics' Jetson Docker image is the fastest path.
   `robot/detect.py` device becomes `cuda` (currently `mps`/`cpu` — one line).
4. `onnxruntime` (CPU) aarch64 wheel for fastembed + onnx-asr — **verified on PyPI** (1.27.0
   download-tested); pin what you test. Note: `onnxruntime-gpu` has **no** aarch64 PyPI wheel —
   CUDA-accelerated ONNX needs NVIDIA's Jetson AI Lab index or a source build. CPU ONNX is fine
   for the embedders at demo cadence.
5. `pyobjc-core` is macOS-only (no Linux wheels) — now marked `sys_platform == 'darwin'` in
   pyproject so `uv sync` doesn't fail on Jetson; the `objc` import in `detect.py` already
   falls back to `nullcontext` off-macOS. Re-run `uv lock` after the marker change.
6. `sounddevice` stays only for laptop-mic dev use (`portaudio19-dev` on Linux if kept);
   the demo path records on the phone.
7. **Headless UI**: bind the view server to `0.0.0.0` (or add `--host`) instead of `127.0.0.1`,
   and add touch buttons on the page for T/A/R/Q — at a booth or shoot the UI runs on a phone/iPad
   on the robot's network (hotspot on the Jetson works offline). The "MEMORY WRITTEN" card is
   filmed evidence, so the companion view is contract-required. **Phone mic**: serve self-signed
   HTTPS (stdlib `ssl`; iOS requires a secure context for `getUserMedia` — tap through the cert
   warning once), record raw PCM via WebAudio, POST a WAV to `/audio`, feed the existing
   transcribe path. Hold-to-talk buttons replace server-side VAD. See PLAN-companion-view.md.
8. DotStar/APA102 over SPI: enable SPI on the 40-pin header (`jetson-io`), then `spidev` or
   Adafruit Blinka + `adafruit_dotstar`. No extra board. Cap brightness (see parts table).
9. TensorRT export of YOLOE is an optimization, not a gate — do it last, if at all.
