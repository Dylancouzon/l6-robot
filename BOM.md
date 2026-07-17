# Bill of Materials

This BOM covers the L6 memory robot demo: a small frosted shell with a Jetson, one camera, and optional addressable LEDs. It is meant for course filming and booth demos, not long-term unattended use.

![Concept render](assets/robot-concept.png)

## Full Build

Recommended target: **Jetson Orin Nano Super 8 GB**.

| # | Part | Suggested pick | Est. USD | Notes |
|---|---|---|---:|---|
| 1 | Compute | Jetson Orin Nano Super 8 GB dev kit | 249 | Main board. Includes the standard 19 V barrel-jack power supply. |
| 2 | Storage | 256 GB NVMe SSD, M.2 2280 | 30 | Use NVMe for boot and model load speed. Avoid microSD for the full build. |
| 3 | Camera | Logitech C920-class UVC USB webcam | 50 | Works with `cv2.VideoCapture` on Jetson. A compact UVC module can make the front lens cleaner. |
| 4 | LED field | APA102 / DotStar strand, 30-60 pixels | 12 | Optional visual effect. Drive from Jetson SPI. Keep brightness low if powered from the header. |
| 5 | Level shifter | 74AHCT125 | 2 | Optional. Use if the LED strand is unreliable with 3.3 V logic. |
| 6 | Battery power | 65 W USB-C PD bank + 20 V trigger cable | 55 | Optional. Use only when the unit needs to look untethered on camera. Test under detector load. |
| 7 | Enclosure | 3D-printed PLA/PETG shell | 6 | Frosted upper shell for LEDs, opaque vented base for the Jetson and fan. |
| 8 | Hardware | Heat-set inserts, M2/M3 screws, JST leads, short USB cables | 15 | Small build hardware and wiring. |
| | | **Required total** | **~$362** | Excludes optional battery and level shifter. |

## Hardware Notes

- Keep the Jetson low in the shell so it acts as the weighted base.
- Leave real intake and exhaust paths. The shell should look soft, but it cannot be sealed.
- Use a UVC USB camera unless you already know you want the CSI camera driver work.
- APA102/DotStar LEDs are simpler than strict-timing LED strips because they use clock and data over SPI.
- If LEDs are powered from the Jetson 5 V header, cap brightness. A 60-pixel strip at full white can pull more current than the header should supply.
- For brighter LEDs, use a separate 5 V buck supply and share ground with the Jetson.
- The phone/tablet is the mic, screen, and control surface. The robot does not need built-in audio or a display.

## Lightweight Build

Experimental only. The software should run on a Raspberry Pi 5 (8/16 GB) with a USB webcam, but CPU-only inference means lower frame rates and the detector needs separate tuning for resolution, cadence, and model size. Not a scoped demo target.

## Model Choices

The full build currently targets the course stack: YOLOE-11L-seg, CLIP ViT-B/32, Nomic v1.5, Whisper-base, and Qdrant Edge.

If the Jetson version moves to a closed COCO detector for better TensorRT support, pre-check every demo object. Props like mugs, bottles, backpacks, and books are safer than unusual objects that may not be detected.

## Jetson Software Checklist

Verify this path before buying extra hardware:

1. Install JetPack 6 on the Jetson. Flashing usually requires an x86 Ubuntu host with NVIDIA SDK Manager.
2. Use the `qdrant-edge-py==0.7.2` aarch64 wheel.
3. Install PyTorch and Ultralytics from NVIDIA's JetPack-compatible packages or container images, not stock desktop CUDA wheels.
4. Keep FastEmbed and `onnx-asr` on CPU ONNX Runtime unless a tested Jetson GPU path is added.
5. Keep `pyobjc-core` macOS-only in `pyproject.toml`.
6. Keep `sounddevice` for laptop development. The demo interface should use phone/tablet audio upload.
7. Serve the browser UI over HTTPS when using a phone mic. Mobile browsers require a secure context for `getUserMedia`.
8. Enable Jetson SPI before driving APA102/DotStar LEDs.
9. Treat TensorRT export as an optimization step after the base demo is working.

## Upgrade Path

The Jetson Orin Nano 8 GB is enough for the current demo stack. If a future version adds a larger image embedder, larger text embedder, or local VLM, the straightforward upgrade is a Jetson Orin NX 16 GB.
