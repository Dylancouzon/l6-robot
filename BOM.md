# Bill of materials

Four things to buy, plus filament. No screws, no inserts.

![L6 robot](assets/robot-render.gif)

| # | Part | Pick | Est. USD | Notes |
|---|---|---|---:|---|
| 1 | Compute | Jetson Orin Nano Super 8 GB dev kit | 249 | Comes with the 19 V barrel power supply. Stands on edge inside the shell and acts as the weight in the base. Measure its height before printing (see [hardware/README.md](hardware/README.md)). |
| 2 | Storage | 256 GB NVMe SSD, M.2 2280 | 30 | Boot and load models from NVMe. Don't use microSD. |
| 3 | Camera | 37 x 37 mm USB (UVC) camera board, ~100 degree lens | 60 | Works with `cv2.VideoCapture`, no drivers. It has to fit the printed tray: **37 x 37 mm board, 1.4 to 1.8 mm thick, connectors on the back, lens barrel no more than 27 mm long racked all the way in, and nothing around the barrel wider than 26.5 mm across.** Measure yours against those numbers with calipers. The Arducam IMX291 (B0200) is the module the design was drawn around. |
| 4 | Enclosure | ~300 g PLA in white, charcoal and red | 8 | Six printed parts. Files and instructions in [hardware/](hardware/). |
| | | **Total** | **~$347** | |

Also needed: one zip tie, for strain relief on the camera cable, and a drop of glue for the eye visor.

## Notes

- Use a USB (UVC) camera unless you want to take on CSI driver work. A desk webcam like a C920 is fine for development on a laptop, but its housing does not fit the shell.
- The shell is vented on purpose. Don't tape the vents shut or block the exhaust at the back.
- The phone or tablet is the microphone, screen, and control surface. The robot has no speaker and no display.
- The Orin Nano 8 GB is enough for the current stack. A later version with a bigger image encoder, a bigger text encoder, or a local VLM would want an Orin NX 16 GB.

Software setup for the Jetson is in [README.md](README.md) under "Running On A Jetson Orin Nano" and "Headless Appliance".
