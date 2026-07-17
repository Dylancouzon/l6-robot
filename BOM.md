# Bill of materials — the memory robot

One self-contained demo unit: a ~140 mm frosted "pebble" — Jetson inside as a weighted, vented base, a single camera "eye," and addressable LEDs across the shell that flicker like an HNSW graph being walked. Booth/talk prop, single build. One board drives the whole unit: the LEDs run on SPI (APA102/DotStar) straight off the Jetson's 40-pin header.

![Concept render](assets/robot-concept.png)

## Parts

| # | Part | Default pick | ~USD | Notes / watch out |
|---|------|--------------|-----:|-------------------|
| 1 | Compute | **Jetson Orin Nano Super 8 GB dev kit** | 249 | Secured. aarch64 + CUDA for YOLOE; 8 GB fits the stack (~2 GB peak) with headroom. Ships with its 19 V barrel-jack PSU. |
| 2 | Storage | 256 GB NVMe SSD (M.2 2280) | 30 | Boot and model loads from microSD are painfully slow; NVMe is the standard Jetson setup. |
| 3 | Camera ("eye") | Logitech C920-class USB (UVC) | 50 | Plug-and-play with `cv2.VideoCapture` on Jetson. Avoid CSI ribbon cams — they need extra driver setup for no demo benefit. A compact USB module (e.g. Arducam UVC) looks better as the lens if you want to shrink the eye. |
| 4 | LED field | Adafruit DotStar / APA102 strand (~30–60 px) | 12 | The graph-traversal shimmer. SPI (clock + data) off the Jetson header — reliable, no timing tricks. **Power budget**: fed from the header's 5 V rail, so brightness must stay capped (~25% / ambient) — 60 px at full white pulls 3.5 A and will brown out the rail. For a brighter look, add a dedicated 5 V buck feed, not more header current. |
| 5 | Level shifter (optional) | 74AHCT125 | 2 | Only if the strand misbehaves at 3.3 V logic; shifts SPI to 5 V. |
| 6 | Untethered power (optional) | **65 W-class** USB-C PD power bank + PD-trigger (20 V) → barrel cable | 55 | Only for a wireless "robot" look on camera; the included 19 V PSU covers any bench or booth-with-outlet shoot. 20 V is in spec — the DC jack accepts 7–20 V. Size at 65 W: MAXN is 25 W module-only, plus camera, NVMe and LEDs. Must sustain 20 V without sagging or auto-sleeping at partial load; test under YOLOE load before any shoot. |
| 7 | Enclosure | 3D-printed shell, matte PLA/PETG | 6 | Frosted/translucent upper for the LEDs, opaque vented base for the Jetson + fan. Matte finish — glossy throws glare on camera. |
| 8 | Fasteners + wiring | M2/M3 heat-set inserts, screws, JST leads, short USB cables | 15 | — |
| | | **Total** | **~$362** | Required parts only; +$57 with the optional power bank and level shifter. |

## Lightweight build (~$145, experimental)

A second tier on a Raspberry Pi 5 instead of the Jetson — everything runs on CPU. Treat it as a learning build for anyone with a Pi on hand; it's unverified, and the detector still needs adapting (see the build-tier note in the README).

| # | Part | Pick | ~USD |
|---|------|------|-----:|
| 1 | Compute | Raspberry Pi 5 8 GB | 80 |
| 2 | Cooling | official Active Cooler | 5 |
| 3 | Power | 27 W USB-C PD PSU | 12 |
| 4 | Storage | 256 GB A2 microSD | 12 |
| 5 | Camera | generic UVC webcam | 20 |
| 6 | Enclosure + wiring | 3D-printed shell, fasteners | 16 |
| | | **Total** | **~$145** |

Qdrant Edge (the aarch64 wheel covers the Pi) and the CPU-ONNX embedders run here as-is, the same path glasses_x_edge uses. The detector is the open part: YOLOE prompt-free wants a GPU, so this tier drops to a smaller CPU variant like `yolo11n-seg` at lower resolution and frame rate. LEDs are optional, as on the full build.

## Headroom (larger models)

The 8 GB Orin Nano runs the current stack (YOLOE-11L-seg + CLIP ViT-B/32 + Nomic v1.5 + whisper-base) comfortably. If a future version needs a bigger CLIP (ViT-L), a larger text embedder, or a concurrent local VLM, the drop-in upgrade is a **Jetson Orin NX 16 GB** — same JetPack image (zero code changes), ~2× RAM and TOPS.

## LED field

The shimmer is decoration, not a live readout, so it sits outside the honest-claim contract's "every score on screen is live" rule (see `CLAUDE.md`). To make it honest, gate each hop on the real recognize loop, which already fires ~continuously.

## Software port checklist (verify before buying anything else)

1. `qdrant-edge-py==0.7.2` aarch64 wheel — **verified on PyPI** (`manylinux_2_28_aarch64`, download-tested; JetPack 6 / Ubuntu 22.04 satisfies it). The one dependency with no substitute is covered.
2. **Flashing needs an x86 Ubuntu host** with SDK Manager to put JetPack 6 on the NVMe (or bootstrap via microSD, then move). Zero-cost, but it's the first step.
3. PyTorch + ultralytics from **NVIDIA's JetPack wheels**, not stock pip — the stock `uv.lock` resolves torch with sbsa/x86 CUDA deps that don't target Jetson's iGPU. Ultralytics' Jetson Docker image is the fastest path. `robot/detect.py` device becomes `cuda` (one line, currently `mps`/`cpu`).
4. `onnxruntime` (CPU) aarch64 wheel for fastembed + onnx-asr — **verified on PyPI** (1.27.0, download-tested); pin what you test. `onnxruntime-gpu` has **no** aarch64 PyPI wheel, but CPU ONNX is fine for the embedders at demo cadence.
5. `pyobjc-core` is macOS-only — marked `sys_platform == 'darwin'` in pyproject so `uv sync` doesn't fail on Jetson (the `objc` import in `detect.py` already falls back to `nullcontext`). Re-run `uv lock`.
6. `sounddevice` stays only for laptop-mic dev use (`portaudio19-dev` on Linux if kept); the demo records on the phone.
7. **Headless UI + phone mic**: bind the view server to `0.0.0.0` (or `--host`) and add touch T/A/R/Q buttons — the UI runs on a phone/iPad on the robot's network (Jetson hotspot works offline). Serve self-signed HTTPS (iOS requires a secure context for `getUserMedia`), record PCM via WebAudio, POST a WAV to `/audio`. The "MEMORY WRITTEN" card is filmed evidence, so this view is contract-required. See PLAN-companion-view.md.
8. DotStar/APA102 over SPI: enable SPI on the header (`jetson-io`), then `spidev` or Adafruit Blinka + `adafruit_dotstar`. Cap brightness (see parts table).
9. TensorRT export of YOLOE is an optimization, not a gate — do it last, if at all.
