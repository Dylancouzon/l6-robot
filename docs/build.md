# Build Your Own Robot

Build the robot in stages. Test the camera and software before printing or gluing anything, then add the phone and automatic startup after recognition works. This order keeps hardware, camera, and network problems separate.

## Build Checklist

- [ ] Run the application with a webcam.
- [ ] Install and test it on the Jetson.
- [ ] Print and assemble the enclosure.
- [ ] Set the camera rotation and crop.
- [ ] Calibrate recognition in the final lighting and position.
- [ ] Test the phone interface.
- [ ] Enable automatic startup and the robot's Wi-Fi network.

## 1. Gather the Hardware

Use the [Hardware Shopping List](hardware-parts.md) to buy or provide every required component. It covers the Jetson, storage, camera, phone or tablet, filament, and assembly supplies.

Buy and test the electronics before printing. The enclosure was designed around the Arducam IMX291 (B0200), and substitute cameras must meet the listed dimensions.

## 2. Test the Software First

Complete [Get Started](getting-started.md) on any available Windows, macOS, or Linux computer. This confirms the application flow before the Jetson and printed parts add more variables.

## 3. Prepare the Jetson

Install NVIDIA's supported JetPack image on the NVMe drive, complete the first-boot account setup, and connect the Jetson to the internet. The reference unit was validated with Ubuntu 24.04, JetPack 7.2, and Jetson Linux R39.2.

Install `uv`, clone this repository as your normal user, and set up the environment:

```bash
git clone https://github.com/Dylancouzon/l6-robot.git qdrant-edge-memory-robot
cd qdrant-edge-memory-robot
uv sync
cp .env.example .env
```

Connect the camera and start the application before assembling the enclosure:

```bash
uv run python -m robot.app
```

Confirm that the camera opens, an object receives a stable box, and **TEACH** records your voice. The first start downloads about 1.5 GB of model files.

## 4. Print and Assemble the Enclosure

Follow [Print and Assemble the Enclosure](hardware.md). Keep these checkpoints in mind:

- Measure a substitute camera before printing.
- Test the camera, cable, and focus before inserting the visor.
- Do not glue the visor until the live feed is upright, focused, and free of blocked edges.

The camera sits upside down in the reference enclosure. Set this value in `.env`:

```dotenv
CAMERA_ROTATE=180
```

If the lens leaves a black rim, use `FRAME_CROP` as described in `.env.example`.

## 5. Calibrate the Final Camera View

Camera distance, lighting, focus, rotation, and crop all change the image vectors. Finish the physical setup first, then follow [Calibrate the Camera](calibration.md).

Do not carry memories from an upside-down or differently cropped test setup into the final configuration. Start clean after the image is final:

```bash
uv run python -m robot.app --reset
```

Teach each object from two or three angles and verify it against the live scene, not only separate test photos.

## 6. Add the Phone Interface

Run the application on the network:

```bash
uv run python -m robot.app --host 0.0.0.0
```

Open the printed HTTPS address on the phone. Follow [Use a Phone](phone.md) for the certificate and microphone permissions.

## 7. Make the Robot Self-Contained

Only after the previous steps work, follow [Set Up a Headless Jetson](appliance.md). The setup script:

- Starts the application at boot and restarts it after a crash.
- Creates a dedicated `qdrant-memory` Wi-Fi network.
- Downloads the models while internet access is still available.
- Enables SSH and disables the desktop to save memory.

The script detects the account and repository path used to run it, so you do not need the original project's username or directory layout.

## Completion Check

A finished build should pass each of these checks:

1. Power on the Jetson without a keyboard or display.
2. Join the robot's Wi-Fi network from the phone.
3. Open `https://10.42.0.1:8765` and see a live feed.
4. Teach an object from the phone.
5. Show it again and see the taught name.
6. Ask where it was last seen.
7. Restart the robot and confirm the memory remains.
