# l6-robot — builder instructions

Instructor-side demo app for L6 of the DLAI course "Building On-Device AI Memory with Qdrant Edge" (course repo: `../on-device-memory-course`; L6 design docs live in its `.build/design/L6/`). Never shipped to students. Also Dylan's standing booth/talk demo. A continuously-running memory loop on camera + mic: capture → detect → embed → match → teach.

## The honest-claim contract (hard rules)

The course says on camera "the robot's memory code mirrors what you wrote." Every rule below exists to keep that true:

- Stack is exactly the course's: **qdrant-edge-py 0.7.2** (pinned), one `EdgeShard`, named vectors `text` 768 (Nomic v1.5) / `image` 512 (CLIP ViT-B/32), both via FastEmbed; **whisper-base** via onnx-asr. `robot/models.py` mirrors the course's `.build/utils/` loaders — keep them aligned.
- `RECOGNIZE_THRESHOLD = 0.80` default, same nearest-match ≥ threshold check as L5. It's a calibration knob (`--threshold`), not scripture — demo quality rules. If live calibration moves it, the course's L5 must be edited to the same number (coherence, not the digit, is the contract).
- Teach-by-voice writes **one point carrying both named vectors** plus transcript and metadata in the payload. The UI's "MEMORY WRITTEN — vectors: image + text" card is filmed evidence (shotlist shot 2) — don't remove it.
- Detector labels are discarded: detection finds *a thing*, memory tells it *which* thing. The one use of class names is person/body-part suppression. Never auto-name objects from YOLOE classes.
- Every score on screen is a live value. **No fabricated output anywhere.** A sync/fleet feature exists only if actually wired.
- V2 fleet rule: **append-only**. Points are pushed and pulled, never merged, folded, or decayed. If a change needs consolidation logic, it belongs in memory-fleet, not here.

## Architecture

- `robot/models.py` — Nomic / CLIP / Whisper loaders (lru_cache singletons), `warm_up(progress)`.
- `robot/memory.py` — the Edge shard: `teach` (both vectors), `remember_sighting` (image only), `recognize` (filter `kind=taught`, nearest vs threshold), `day_recall` (both spaces, `ts` range filter, seen/heard never merged), `reopen` (the offline-reboot beat).
- `robot/detect.py` — YOLOE prompt-free + BoT-SORT + stability/cadence gate + masked crops. All constants field-tuned in memory-fleet; `../hive-mind` is the local reference checkout.
- `robot/audio.py` — sounddevice push-to-talk with VAD auto-stop, silence guard, label parsing.
- `robot/core.py` — `Robot`: the loop minus the camera. Both front ends drive it, so the smoke test exercises what the shoot records.
- `robot/app.py` — live mode (browser view at `http://127.0.0.1:8765`, MJPEG stream, T/A/R/Q keys in the tab) and headless replay (`--source dir|video`).
- `smoke_test.py` — the contract: teach two views + WAV → held-out ≥ threshold → foreign stays unknown → day question → reboot → all repeats. Run it after every change: `uv run python smoke_test.py`.

## Field-learned decisions — do not relitigate, recalibrate

- **Browser view, not an OpenCV window.** OpenCV's macOS windowing breaks on multi-monitor setups (vanishes, shrinks, sticks fullscreen). The web page is the fix, same as memory-fleet.
- **sounddevice, not ffmpeg/avfoundation.** ffmpeg failed to open two different USB audio devices on Dylan's machines. Recording uses the device's native rate, resamples to 16 kHz. `rms == 0` means the terminal app lacks macOS mic permission — tell Dylan, don't debug the code.
- **Perception constants come from memory-fleet** (conf 0.30, area band 0.0008–0.20, masked crops with gray fill, sticky person suppression, 3-frame stability, 2 s requery). Tune live via `--conf` / `--max-area` / `--threshold`; don't hand-edit constants on a hunch — the 0.20 area cap is what stops it tracking people/torsos, and the mask fill is what makes recognition survive hands and backgrounds.
- **One unknown at a time.** All recognized objects display; only the most salient unknown (size × centrality) is shown, written as a sighting, and teachable. Keeps the view and the day-recall clean.
- **Whisper hallucinates on silence** ("Thanks for watching!") — the RMS silence guard before transcription is load-bearing, keep it.
- **Voice actions run off the main loop** (`busy` flag, key-queue drain). Recording on the main loop froze the feed and serially queued repeat presses.
- Weights path resolves against the repo root, not cwd (`yoloe-11l-seg-pf.pt`, 70 MB, gitignored; copy from `../hive-mind` or let ultralytics download).
- Re-teaching an object adds a second point; the nearest view's note wins. Known ceiling, ponytail-marked in `memory.py`; fold-by-label only if it ever matters.

## Commands

```bash
uv sync
uv run python -m robot.app                    # live; knobs: --threshold --conf --max-area --camera --data
uv run python -m robot.app --source testdata  # headless replay
uv run python smoke_test.py                   # must pass before any commit
uv run python -c "import sounddevice; print(sounddevice.query_devices())"  # pick MIC_DEVICE if default is wrong
```

## Working agreements

- Local git only: **no remote, no push, no publishing** without Dylan's explicit ask. Commit per working milestone.
- Ponytail applies: smallest app that produces every shotlist beat honestly. No config systems, no plugin architecture, no speculative fleet code.
- UI: high contrast, readable from across a room, no dark generic-terminal look — these demos run at booths and on camera.
- Dogfooding by cheap-model subagents goes in `dogfood/` (gitignored), scripted against the `Robot` class; they never edit `robot/`.
- Fixture images in `testdata/` are Wikimedia Commons — keep `CREDITS.json` 1:1 if any are added.

## What's next (state at last session)

1. **Single-unit definition of done**: file-mode smoke test passes; mic works live (permission + sounddevice fixed). Still to record: one clean live session — teach by voice → recognize from a new angle → day question → offline reboot — then reconcile the course's `.build/design/L6/SCRIPT.md` (marked PROVISIONAL) against real values.
2. **V2 — fleet intro**: full build plan and design rules in README.md ("V2: an introduction to the fleet"). Prerequisite refactor (thumbnails → payload base64) is listed there. Check current Edge sync API on skills.qdrant.tech before writing transport code.
3. **Jetson port**: hardware scoping table + software checklist in README.md. Verify the qdrant-edge-py aarch64 wheel before buying peripherals.
