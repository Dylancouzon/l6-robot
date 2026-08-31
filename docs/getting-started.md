# Get Started

This guide runs the complete application with an ordinary computer, webcam, and microphone. It is the fastest way to verify the software before building the Jetson enclosure.

## What You Need

- 64-bit Windows (x86), macOS, or Linux.
- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/).
- A webcam and microphone.
- Enough disk space for about 1.5 GB of downloaded model files.

You do not need a Qdrant server, cloud account, API key, or printed enclosure.

## Install the Project

From the repository root:

```bash
uv sync
cp .env.example .env
```

In Windows PowerShell, use `Copy-Item .env.example .env` instead of `cp`.

`uv sync` creates an isolated environment and installs the versions in `uv.lock`. The `.env` file holds settings that depend on your camera. Keep the defaults for the first run.

## Start the Robot

```bash
uv run python -m robot.app
```

The application opens `http://127.0.0.1:8765`. Open that address manually if your browser does not appear. The first start downloads the detector, image encoder, text encoder, and speech model, so it takes longer than later starts.

On Windows, allow Python through Windows Defender Firewall if prompted. Local use at `127.0.0.1` does not require network access; the firewall permission is only needed when serving the interface to another device with `--host 0.0.0.0`.

If the wrong camera opens, stop with Ctrl-C and try another index:

```bash
uv run python -m robot.app --camera 1
```

## Teach and Recall an Object

1. Hold one object near the center of the frame until its box becomes steady. The thickest box is the current target.
2. Hold **TEACH** or the `T` key, say “This is my mug,” and release it.
3. Turn the object and teach it two more times. Each teaching adds another view instead of replacing the first.
4. Move the object out of view, then show it again. Its box should display the name you taught.
5. Open **MEMORY** to inspect the taught views and later sightings.
6. Hold **ASK** or the `A` key and ask, “When did you last see my mug?”

Speak a short sentence instead of a single noun. Whisper has more context to transcribe, and Qdrant can later match questions against the full sentence.

## Controls

| Control | Action |
|---|---|
| `T` or **TEACH** | Teach the focused unknown object. |
| `A` or **ASK** | Ask where an object was seen. |
| `M` or **MEMORY** | Review and edit saved objects. |
| `Q` or **IGNORE** | Hide the focused unknown object. |
| `F` | Forget the focused recognized object. |
| `R` | Close and reload the memory shard from disk. |
| Ctrl-C | Stop the application. |

Use the memory view for deliberate deletion. Camera focus can move between objects, so `F` is best kept as a development shortcut.

## Check the Pipeline Without a Camera

The bundled images can exercise detection, cropping, image embedding, and memory lookup:

```bash
uv run python -m robot.app --source testdata/
```

With an empty memory, the images correctly print as `UNKNOWN`. This is a software smoke test, not a recognition accuracy test.

## Where Data Lives

The local Qdrant Edge shard and its thumbnails live in `edge-data/`. This directory is ignored by Git. Start once with an empty shard by adding `--reset`:

```bash
uv run python -m robot.app --reset
```

`--reset` deletes the selected shard before startup. Do not use it if you want to keep what the robot learned.

## Common Setup Issues

### The Robot Matches Too Many Objects

Your camera needs a higher recognition threshold. Follow [Calibrate the Camera](calibration.md). A score of `0.90` is a cosine similarity threshold, not “90% confident.”

### The Robot Misses Different Angles

Teach the same object two or three times while turning it. Recognition searches for the nearest taught view, so each additional view covers another appearance.

### The Wrong Microphone Is Used

List the available devices:

```bash
uv run python -c "import sounddevice; print(sounddevice.query_devices())"
```

Set `MIC_DEVICE` in `.env` to a device name or numeric index, then restart the application.

### A Phone Cannot Use Its Microphone

Phone browsers require HTTPS for microphone access. Follow [Use a Phone](phone.md) instead of opening the laptop's plain HTTP address.

### The Camera Feed Is Upside Down or Has a Black Rim

Set `CAMERA_ROTATE` or `FRAME_CROP` in `.env`. Read the comments in [`.env.example`](../.env.example) before teaching objects because changing rotation invalidates existing taught views.

## Next Steps

- Follow [Build Your Own Robot](build.md) to move the tested application to a Jetson.
- Read [Understand the Architecture](architecture.md) to connect the course notebooks to the source code.
