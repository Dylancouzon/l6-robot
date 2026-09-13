# Print and Assemble the Enclosure

![The Qdrant Edge Memory Robot: a printed desktop enclosure with a camera behind its visor](../assets/robot-render.gif)

The enclosure holds a Jetson Orin Nano and a USB camera. Your phone provides the screen, microphone, and controls.

Before you print, buy or provide every item in the [Hardware Shopping List](hardware-parts.md). The printable files contain the enclosure only, not the electronics.

## Check the Camera Fit

The enclosure was designed around the Arducam IMX291 (B0200). Before printing for a different camera, confirm that it meets the dimensions in the [Hardware Shopping List](hardware-parts.md).

Measure the physical camera when possible. The glued visor makes a later camera change difficult.

## Print the Parts

The ready-to-print files are in [`hardware/`](../hardware/):

| File | Color | Parts |
|---|---|---|
| `white_plate.3mf` | White | Shell |
| `charcoal_plate.3mf` | Charcoal | Keel, camera clamp, eye visor, antenna |
| `red_plate.3mf` | Red | Antenna tip |

The separate `.stl` files contain the same six parts.

Use these settings:

- PLA, 0.4 mm nozzle, 0.2 mm layers, three walls, and 15% infill.
- Add a brim to the shell, keel, visor, and antenna.
- Add support only below the keel's camera head. Use a dense support interface.
- Check the slicer preview of the camera window. Its sloped roof must bridge cleanly so it does not block the camera connectors.

The largest plate is 177 × 176 mm. Printing all parts takes about a day and a half.

## Assemble the Robot

1. Place the Jetson on edge in the keel, with its ports toward the rear and its power socket at the bottom. Insert one end first, swing the other end down, and press it onto the four pads. Do not connect power yet.
2. Feed the camera cables through the camera-head window. Slide the camera board straight in from the front until it reaches the back stop.
3. Place the clamp around the lens and push its arms into the two channels until both clips click.
4. Turn the camera focus fully inward.
5. Lift the assembly through the bottom of the shell. Twist it 18° to the alignment mark.
6. Connect power through the bottom opening. Route the cable through the rear slot.
7. Start the application and adjust the camera focus through the bottom opening.
8. Confirm that the image is upright and the enclosure does not block the view.
9. Press the eye visor straight in and secure it with one drop of glue.
10. Insert the antenna at the top and turn it one quarter turn.
11. Use the zip tie to keep the camera cable against the keel.

Glue the visor last. Removing it later may break it.

## Change the Model

Import [`claude-design.html`](../hardware/claude-design.html) into Claude Design to edit the model, or open it directly in Chrome. Wait about 45 seconds for the model and audit to finish, then export new plates. Do not print a model that reports an audit failure.
