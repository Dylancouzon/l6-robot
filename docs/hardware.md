# The Robot Body

![The L6 robot: a printed shell with a camera behind its visor and an antenna on top](../assets/robot-render.gif)

A 130 x 190 mm printed shell with a Jetson standing on edge inside it, one USB camera behind the eye, and an antenna on top. It is a filming and booth prop rather than a consumer device.

- **Jetson Orin Nano Super 8 GB**, low in the shell, where its weight keeps the robot upright.
- **A 37 x 37 mm USB (UVC) camera** as the eye. The printed tray is cut to that size, so measure yours before printing.
- **No microphone, speaker, or screen.** Your phone browser is the whole interface. Spoken answers use macOS `say` when you run this on a Mac; on the Jetson the same sentence appears on the panel.
- **Six printed parts, no screws.** A twist lock, two snap clips, one zip tie, and a drop of glue on the visor.

A Raspberry Pi 5 with a USB webcam should run the software, but CPU-only inference makes it slow and it does not fit this shell. It is not a tested target.

## What To Buy

Four things, plus filament. No screws, no inserts.

| # | Part | Pick | Est. USD | Notes |
|---|---|---|---:|---|
| 1 | Compute | Jetson Orin Nano Super 8 GB dev kit | 249 | Comes with the 19 V barrel power supply. Stands on edge inside the shell and acts as the weight in the base. Measure its height before printing. |
| 2 | Storage | 256 GB NVMe SSD, M.2 2280 | 30 | Boot and load models from NVMe. Don't use microSD. |
| 3 | Camera | 37 x 37 mm USB (UVC) camera board, ~100 degree lens | 60 | Works with `cv2.VideoCapture`, no drivers. It has to fit the printed tray, so check it against the measurements below. The Arducam IMX291 (B0200) is the module the design was drawn around. |
| 4 | Enclosure | ~300 g PLA in white, charcoal and red | 8 | Six printed parts, in [hardware/](../hardware/). |
| | | **Total** | **~$347** | |

Also needed: one zip tie for strain relief on the camera cable, and a drop of glue for the eye visor.

Use a USB (UVC) camera unless you want to take on CSI driver work. A desk webcam like a C920 is fine for developing on a laptop, but its housing does not fit the shell. The shell is vented on purpose, so do not tape the vents shut or block the exhaust at the back.

The Orin Nano 8 GB is enough for the current stack. A later version with a bigger image encoder, a bigger text encoder, or a local vision-language model would want an Orin NX 16 GB.

## Print It

Six parts, no screws. Budget about a day and a half of printing. The files are in [hardware/](../hardware/), as three plates, one per filament colour:

| File | Colour | Parts on it |
|---|---|---|
| `l6b1_white_plate.3mf` | white | shell |
| `l6b1_charcoal_plate.3mf` | charcoal | keel, camera clamp, eye visor, antenna |
| `l6b1_red_plate.3mf` | red | antenna tip |

Each plate is already laid out and the biggest is 177 x 176 mm, so it fits any common bed. These are plain 3MF files with no printer-specific settings in them, so PrusaSlicer, Orca, and Cura read them as happily as Bambu Studio does. The six `l6b1_*.stl` files are the same six parts, one per file, for a toolchain that will not take a 3MF.

To change the model, open `l6-bot-v54.html` in Chrome. It builds the shape in the browser, which takes about 45 seconds, and the tab looks frozen while it works. Then use the export buttons to write new plates. An audit box in the corner re-checks the model every time the page loads; if it reports a failure, do not print.

### Measure Before You Print

Get out the calipers and check three things.

- **Jetson height**, nominal 103 mm. If yours differs, change `KIT_LIFT` near the top of the HTML and export again. This is what makes the Jetson a tight fit instead of a rattle.
- **Camera board**: 37 x 37 mm, 1.4 to 1.8 mm thick, connectors on the back face.
- **Lens barrel, racked all the way in**: 27 mm at most from the front of the board, and nothing around it wider than 26.5 mm across.

Get a camera number wrong and the board will not go in. There is no second chance, because the visor is glued.

### Printer Settings

Bambu P1S (256 mm bed) or anything similar. PLA, 0.4 mm nozzle, 0.2 mm layers, 3 walls, 15% infill. **Use the same profile for every part**, with no per-part overrides.

- **Supports: one patch only**, under the keel's camera head, about 108 mm up. Turn on a dense support interface. It holds up a thin 40 mm pillar beside the camera window; if that patch comes loose, the roof of the window loses its anchor.
- **Brim** on the shell, keel, visor, and antenna.
- The keel is the big one: 160 mm tall, about 103 cm3. A failed print costs most of a day, so watch the first layers.
- Look at the slicer preview of the keel's camera window. Its roof is a 32 mm bridge tilted 8.6 degrees, and slicers often read that as an overhang and let it sag. Sagging there eats the clearance over a connector.

## Assemble It

The order matters. Do not skip ahead.

1. **Jetson into the keel**, on edge, ports toward the back, power end down. Put one end in first and swing the other down; it will not go in flat. Press it home onto the four small pads. Leave the power plug off for now.
2. **Camera board in from the front**, along the lens axis, never from above. Feed both cables back through the window in the camera head first, then slide the board straight back until it stops against the frame behind it.
3. **Clamp on**: lens through the open side, arms down the two channels, push until both clips click.
4. **Rack the focus all the way in.**
5. **Lift the whole assembly up through the hole in the bottom of the shell**, then twist 18 degrees to lock. There is a little free play past the lock, so twist to the mark rather than until it stops.
6. **Plug in the power barrel**, reaching in through the bottom hole, and lead the cable out of the slot at the back so it turns down onto the desk.
7. **Set the focus** through the bottom hole.
8. **Glue the eye visor in last**, pressed straight in. One drop. This step is permanent.
9. **Antenna into the hole on top**, quarter turn.

Zip tie the camera cable to the keel wherever it wants to bunch up.

## Taking It Apart Again

The camera board comes out: pry the clamp up at the notch in the middle until the clips let go, then slide the board straight out the front. The Jetson lifts out after that.

The visor does not come out without breaking it. Everything above the camera is a one-way trip.

## Reference

[hardware/SPEC.md](../hardware/SPEC.md) is the full engineering spec: every dimension, tolerance, and past mistake. Read it before changing the model, not before printing it.
