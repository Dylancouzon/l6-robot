# Understand the Architecture

This repository turns the course's notebook memory into a continuously running device. The central design rule is simple: `robot/device` can import `robot/brain`, but `robot/brain` never imports `robot/device`. You can therefore understand and test the memory logic without a camera, browser, web server, or thread.

## How the Course Maps to the Code

| Course Concept | Implementation |
|---|---|
| Store vectors and payloads in an embedded shard | `Memory` in `robot/brain/memory.py` |
| Search notes by meaning | Nomic text vectors in `robot/brain/models.py` and `Memory.best_taught` |
| Transcribe voice locally | `models.transcribe` with Whisper |
| Teach an object with image examples | `Robot.teach` and `Memory.teach` |
| Recognize with nearest search and a threshold | `Memory.recognize` |
| Calibrate with held-out views | `testdata/verify_scores.py` |

The robot adds object detection, tracking, a live camera loop, persistent sightings, and a phone interface around those same ideas.

## Read the Code in This Order

1. `robot/brain/core.py`: `Robot.process_frame` connects the full recognition loop. `teach` and `ask` are the two voice actions.
2. `robot/brain/memory.py`: the Qdrant Edge schema, writes, recognition query, and recall query.
3. `robot/brain/models.py`: lazy loaders for CLIP, Nomic, and Whisper.
4. `robot/brain/detect.py`: YOLOE detection, BoT-SORT tracking, crop masks, and the stability gate.
5. `robot/device/live.py`: camera and worker threads, button actions, and the bridge to `Robot`.
6. `robot/device/server.py` and `robot/device/page.html`: HTTP routes and the browser interface.
7. `robot/app.py`: command-line setup, live startup, and file replay.

## Recognition Flow

For each stable tracked object:

```text
camera frame
  -> Detector.process
  -> padded object crop
  -> models.embed_crop
  -> Memory.recognize
  -> nearest taught image vector
  -> score compared with RECOGNIZE_THRESHOLD
```

`Detector` discards YOLOE's class label. It uses only the box, segmentation mask, confidence, and tracking ID. That makes teaching a memory operation instead of detector retraining.

`Robot.process_frame` limits memory queries to stable tracks and rechecks them every two seconds. When a track matches, the robot writes one `seen` point for that tracked appearance. It also chooses one prominent unknown as the teach target so background detections do not compete for the button.

## Teaching Flow

```text
TEACH press
  -> save the current unknown crop
  -> record WAV
  -> Whisper transcript
  -> labels.parse_label
  -> CLIP image vector + Nomic text vector
  -> Memory.teach
```

A taught view contains both named vectors. Teaching the same label again adds a point. It does not overwrite the first view. Recognition asks for the closest of all taught views, which is why two or three angles work better than one.

## Recall Flow

```text
ASK press
  -> record WAV
  -> Whisper question
  -> Nomic query vector
  -> choose the taught object
  -> retrieve its latest sightings by time
```

If the question contains an exact taught name, recall narrows candidates to named objects first. This prevents a semantically similar but differently named object from replacing an explicit request. The final answer comes from stored payload fields such as `label`, `ts`, `where`, and `thumb`; no language model generates it.

## Memory Schema

One Qdrant Edge shard lives in `edge-data/`. Points use one of three `kind` values:

| Kind | Vectors | Purpose |
|---|---|---|
| `taught` | `image` and `text` | A named example view and its spoken description. |
| `seen` | `image` | A later sighting of a known object. |
| `ignored` | `image` | A dismissed piece of clutter that should remain hidden. |

The `image` vector has 512 dimensions and uses CLIP. The `text` vector has 768 dimensions and uses Nomic. Both use cosine distance. Payloads hold the human-readable label, transcript, timestamp, location, and thumbnail paths.

The application flushes after each write because disconnecting power is the physical robot's normal shutdown. The `R` control closes and reloads the shard to demonstrate that recall comes from device storage.

## Runtime Boundaries

`LiveApp` keeps camera rendering responsive by separating four jobs:

| Job | Responsibility |
|---|---|
| Main loop | Read frames, draw boxes, and publish JPEG images. |
| Detection worker | Run YOLOE, embed crops, and query memory. |
| Key worker | Process button and keyboard actions in order. |
| Voice worker | Record or receive audio, transcribe it, then teach or ask. |

Shard access is serialized inside `Memory`. Live tracking state is protected by `LiveApp.lock`. The lock order is live state first, then memory.

## Configuration and State

- `.env` contains camera-dependent settings. `robot/config.py` validates and loads them.
- `edge-data/` contains the Qdrant shard, thumbnails, and saved location.
- `~/.cache/fastembed/` contains downloaded embedding models.
- `cert/` contains a generated local HTTPS certificate when a phone connects.

Configuration precedence is: source defaults, `.env`, exported environment variables, then command-line flags where a flag exists.

## Safe Places to Experiment

- Change recognition and detection thresholds in `.env`, then use `testdata/verify_scores.py`.
- Change spoken label parsing in `robot/brain/labels.py`.
- Change recall behavior in `Robot.ask` and the corresponding `Memory` queries.
- Change the browser presentation in `robot/device/page.html` without touching memory.

Run the bundled image replay after changing detection, cropping, encoders, or recognition:

```bash
uv run python -m robot.app --source testdata/
```

Do not test destructive changes against a shard you want to keep. Pass `--data` with a separate directory.
