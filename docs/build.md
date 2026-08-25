# Build the robot body

![The L6 robot](../assets/robot-render.gif)

The enclosure holds a Jetson Orin Nano and a USB camera. Your phone provides the screen, microphone, and controls.

## Parts

| Part | Requirement |
|---|---|
| Computer | Jetson Orin Nano Super 8 GB developer kit with its power supply |
| Storage | 256 GB or larger M.2 2280 NVMe SSD |
| Camera | 37 × 37 mm USB UVC camera board with a lens near 100° |
| Filament | About 300 g of PLA in three colors |
| Small items | One zip tie and a drop of glue |

The enclosure was designed around the Arducam IMX291 (B0200). A different camera must meet all three measurements below:

- Board: 37 × 37 mm and 1.4 to 1.8 mm thick, with connectors on the back.
- Lens length: no more than 27 mm from the front of the board when focused fully inward.
- Lens width: no part around the barrel wider than 26.5 mm.

Measure the camera before printing. The glued visor makes a later camera change difficult.

## Print the parts

The ready-to-print files are in [`hardware/`](../hardware/):

| File | Color | Parts |
|---|---|---|
| `l6b1_white_plate.3mf` | White | Shell |
| `l6b1_charcoal_plate.3mf` | Charcoal | Keel, camera clamp, eye visor, antenna |
| `l6b1_red_plate.3mf` | Red | Antenna tip |

The separate `.stl` files contain the same six parts.

Use these settings:

- PLA, 0.4 mm nozzle, 0.2 mm layers, three walls, and 15% infill.
- Add a brim to the shell, keel, visor, and antenna.
- Add support only below the keel’s camera head. Use a dense support interface.
- Check the slicer preview of the camera window. Its sloped roof must bridge cleanly so it does not block the camera connectors.

The largest plate is 177 × 176 mm. Printing all parts takes about a day and a half.

## Assemble the robot

1. Place the Jetson on edge in the keel, with its ports toward the rear and its power socket at the bottom. Insert one end first, swing the other end down, and press it onto the four pads. Do not connect power yet.
2. Feed the camera cables through the camera-head window. Slide the camera board straight in from the front until it reaches the back stop.
3. Place the clamp around the lens and push its arms into the two channels until both clips click.
4. Turn the camera focus fully inward.
5. Lift the assembly through the bottom of the shell. Twist it 18° to the alignment mark.
6. Connect power through the bottom opening. Route the cable through the rear slot.
7. Adjust the camera focus through the bottom opening.
8. Press the eye visor straight in and secure it with one drop of glue.
9. Insert the antenna at the top and turn it one quarter turn.
10. Use the zip tie to keep the camera cable against the keel.

Glue the visor last. Removing it later may break it.

## Change the model

Open [`l6-bot-v54.html`](../hardware/l6-bot-v54.html) in Chrome. Wait about 45 seconds for the model and audit to finish, then export new plates. Do not print a model that reports an audit failure.
