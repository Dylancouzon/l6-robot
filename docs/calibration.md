# Calibrate the Camera

If you teach one object and half the room starts matching it, the recognition threshold is too low.

The recognition threshold depends on the camera, distance, and lighting. The default will not fit every setup.

## Why Calibration Matters

CLIP cosine similarity measures how closely two vectors point in the same direction. It is not a percentage. Two unrelated crops from the same camera can score between 0.75 and 0.85 because they share the same lighting, sensor, background, and scale. A score of `0.90` does not mean "90% confident." It is simply a value above that common range.

Move the camera farther away and every crop gets smaller and softer. The floor rises, and a threshold that worked at arm's length starts matching the furniture.

## Find a Threshold

```bash
uv run python testdata/verify_scores.py
```

The script crops images with the same code used by the live robot. It prints the score ranges for matching and unrelated objects, the gap between those ranges, and the results at several possible thresholds.

Put a value from the clean range into `.env` as `RECOGNIZE_THRESHOLD`.

To calibrate against your own scene, photograph two or three objects three times each, name the files `<object>_<n>.jpg`, and point the script at them:

```bash
uv run python testdata/verify_scores.py --source ~/my-photos
```

Separate photos of distinct objects score farther apart than a live cluttered scene does. Treat the script's answer as a floor, and expect to raise it against the real thing.

If the gap is negative, the two score ranges overlap and no threshold will separate them. Get closer, add light, or fill more of the frame with the object.

## Other Settings in `.env`

`DETECT_MAX_AREA` drops boxes bigger than a fraction of the frame. A prompt-free detector often proposes walls, desks, and whole rooms.

`DETECT_MIN_AREA` drops small far-away clutter that can steal the unknown box from the thing you are holding up.

`DETECT_CONF` is the detector's confidence floor. Raise it to track less clutter.

## Flags

Use flags to try settings for one run. Save the values that work in `.env`.

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
