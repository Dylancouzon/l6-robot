# Calibrating For Your Camera

**Symptom:** you teach one object and half the room starts matching it, just over the bar.

Nothing is broken. The recognition threshold is a per-camera number, and the shipped default belongs to one specific camera in one specific room.

## Why The Default Is Not Yours

CLIP cosine similarity, a measure of how close two vectors point, does not behave like a percent from 0 to 1. Two unrelated crops from the same camera can routinely score 0.75 to 0.85, because they share lighting, sensor, background, and scale. `0.90` is not "90% confident". It is a point above that floor.

Move the camera farther away and every crop gets smaller and softer, the floor rises, and a threshold that worked at arm's length starts matching the furniture.

## Find Your Number

```bash
uv run python testdata/verify_scores.py
```

The script crops through the same code path the live robot uses and prints three things: the same-object and different-object score ranges, the margin between them, and a threshold sweep showing how many true matches survive at each candidate.

Put a value from the clean range into `.env` as `RECOGNIZE_THRESHOLD`.

To calibrate against your own scene, photograph two or three objects three times each, name the files `<object>_<n>.jpg`, and point the script at them:

```bash
uv run python testdata/verify_scores.py --source ~/my-photos
```

Separate photos of distinct objects score farther apart than a live cluttered scene does, so treat the script's answer as a floor and expect to raise it against the real thing.

If the margin comes out negative, no threshold can work and the crops are the problem. Get closer, add light, or fill more of the frame with the object.

## The Other Knobs In .env

`DETECT_MAX_AREA` drops boxes bigger than a fraction of the frame, which a prompt-free detector proposes freely for walls, desks, and whole rooms.

`DETECT_MIN_AREA` drops the small far-away clutter that keeps stealing the unknown box from the thing you are holding up.

Both are *areas*, so they move as the square of apparent size: going from `0.0008` to `0.001` raises the smallest tracked object by about 12% in width, not 25%.

`DETECT_CONF` is the detector's confidence floor. Raise it to track less clutter.

## Flags

Every per-camera setting has a permanent home in `.env` and a flag that overrides it for one run, which is what you want while calibrating.

| Flag | What it does |
|---|---|
| `--threshold 0.90` | Recognition bar. `.env` `RECOGNIZE_THRESHOLD` |
| `--conf 0.30` | Detector confidence floor. `.env` `DETECT_CONF` |
| `--max-area 0.20` | Biggest detection kept, as a fraction of the frame. `.env` `DETECT_MAX_AREA` |
| `--location "Hotel room"` | Place stamped on this session's memories, which recall reads back |
| `--reset` | Wipe all memories before starting |
| `--camera 1` | Use a different webcam |
| `--host 0.0.0.0` | Serve the view on the network, for a phone or iPad |
| `--advertise 10.42.0.1` | Address to put in the URL and the certificate, for the appliance |
| `--watchdog 30` | Exit if the camera delivers no frame for this long, so a service manager can restart. Off by default, because a closed laptop lid looks exactly like a stalled camera |
| `--source DIR` | Headless replay over images or a video, with no camera |
| `--data DIR` | Shard directory. Point this at a scratch copy when testing |
