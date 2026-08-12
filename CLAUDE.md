# Jetson Deployment Status

Durable handoff for the NVIDIA Jetson deployment. Update it when a test changes a
conclusion or a milestone completes.

**Status at 2026-08-12: solved and shipped.** The robot runs on the Jetson from a
plain `uv` install. No container, no reflash, no source-built PyTorch.

**Second issue, same day: recognition matched everything. Also solved** — the
0.80 threshold was calibrated against a crop distribution the robot never
produces. See "Recognition threshold" below.

**Third issue, same day: the UI stuttered on FORGET/TEACH and the UNKNOWN box
hopped between objects.** Fixed in code and reviewed, **not yet confirmed on the
device by the operator** — see "UI responsiveness" below for what to watch for
and which two traps the review closed.

## Goal and constraints

- Target: Jetson Orin Nano Super 8 GB, Ubuntu 24.04/aarch64, Jetson Linux R39.2 /
  JetPack 7.2, NVMe, original NVIDIA 19 V supply.
- Power mode stays 15 W. Do not change fan controls, over-current limits, or the
  device tree, and do not use `jetson_clocks`/MAXN. None of them are needed.
- Preserve the recognition models and the pinned FastEmbed/onnx-asr versions. The
  threshold (now `0.90`, in `.env`) is calibrated against those exact embeddings
  *and* against the live crop pipeline, so any change to encoders, providers, or
  crops requires a score-parity re-run via `testdata/verify_scores.py`.
- This repo is teaching material for L6 of the course. Prefer the smallest change
  that works; extra machinery costs more here than a few hundred MB.
- Prefer official NVIDIA documentation and staff forum guidance. Do not guess
  wheel URLs.
- Preserve user changes.

## Repository state

- Repository: `/home/qdrant/Documents/github/l6-robot`
- PR #1 (`codex/enable-jetson-cuda`, the Jetson work) is **merged** into `main`.
- PR #2 (the threshold calibration described under "Recognition threshold") is
  **merged** into `main` as of 2026-08-12.
- Branch: `codex/calibrate-recognition-threshold`, reused after that merge and
  now one commit ahead of `main`: the responsiveness work described under "UI
  responsiveness", open as **PR #3**. It touches `robot/app.py`,
  `robot/core.py`, `robot/detect.py`, the README and this file; it adds no
  files and no dependencies.
- `robot.log` is untracked runtime output and is now covered by `*.log` in
  `.gitignore`. `.env` is gitignored too; `.env.example` is tracked and is the
  documentation for every per-camera knob.
- This file is **tracked** as of the threshold PR. Keep it updated in the same
  commit as the change it describes, rather than letting it drift.

## The problem and the fix

The robot updated its view roughly once every 15 seconds. **The cause was startup
memory, not CUDA.**

`warm_up()` eagerly built all four ONNX encoders, and ONNX Runtime gives every
session one spinning thread per core. That reached 3.3 GB and 78 threads before
the camera opened, left ~280 MB available, and pushed 2 GB of the process into
swap — so every teach/ask paged it back in.

Keeping Whisper resident also bought little: transcription costs ~6 s regardless
of residency, so it saved ~3 s of load time for 1.7 GB.

The fix, all in `robot/models.py` plus one call-site change:

- `warm_up()` loads **only the CLIP vision encoder**, the one model the camera
  loop uses per frame. The other three load lazily on first use via `lru_cache`,
  hidden under the button hold.
- `ENCODER_THREADS = 2` with spinning disabled, via FastEmbed's `threads`
  parameter and onnx-asr's `sess_options`. Both are documented public options.
  This is what keeps the feed live while Whisper thinks.
- `Robot.teach()` takes the transcript, not the WAV, so `app.py` runs the slow
  speech-to-text *before* claiming the live-state lock instead of blocking the
  detector for seconds. Removed the then-unused `ask_from_wav()`.

| | Before | After |
|---|---|---|
| Startup RSS | 3.3 GB | 2.3 GB |
| Threads | 78 | 18 (34 after an interaction) |
| Swapped out at startup | 2.0 GB | 0 |

Do not treat swap as the solution. The 8 GB swapfile can stay, but a correct
configuration does not touch it.

## Recognition threshold — why it moved to 0.90

Symptom: teach one object, and almost everything in frame scores 0.80–0.88
against it while the true object sits above 0.90.

**The Qdrant layer was verified clean, empirically, against the live shard.**
Do not re-investigate it: stored image vectors have L2 norm `1.000000` (Qdrant
pre-normalizes on upsert for Cosine, as documented), shard scores match
hand-computed numpy cosine to `1e-7`, and the named vectors are correctly
wired (`image` 512, `text` 768, not swapped).

The cause is that **CLIP cosine similarity has a high floor**, and the 0.80
number came from a harness that didn't exercise the live crop pipeline. The
old `verify_scores.py` cropped large studio photos with no segmentation mask;
the app masks and feeds crops as small as 36x95 px. Ablated on the same
labelled testdata:

| crop condition | same-obj med | cross-obj med | cross max |
|---|---|---|---|
| plain crop (what the old harness measured) | 0.928 | 0.479 | 0.535 |
| + mask gray-fill | 0.920 | 0.472 | 0.611 |
| plain crop, shrunk to live scale | 0.941 | 0.626 | 0.640 |
| mask-fill + shrunk (**what the app does**) | 0.948 | 0.611 | 0.655 |

Resolution is the dominant factor; the gray mask fill mostly fattens the tail.
Live scene crops score higher still than any of these, because they share
camera, lighting and background.

Measured on the live shard (2 taught, 47 sightings), scoring every sighting
against each taught object:

| taught | med | p90 | ≥0.80 | ≥0.90 |
|---|---|---|---|---|
| Mirror (blurry, low-contrast, reflective) | 0.831 | 0.886 | **47/47** | 3/47 |
| Smartphone (sharp, distinctive) | 0.736 | 0.801 | 5/47 | 2/47 |

Sharp knee between 0.80 and 0.85, flat from 0.90 up. Hence `0.90`.

**Known tension, accepted:** on the bundled studio photos, 0.90 drops 2 of 6
same-object pairs (0.887, 0.899). The threshold is justified by the live
measurement, not by the testdata. If recognition starts dropping objects that
turn or relight, 0.88 is the next stop — re-measure, don't guess.

### What changed

- `robot/config.py` (new) reads `.env` via `python-dotenv` (added to
  `pyproject.toml`); `.env.example` documents every knob and `.env` is
  gitignored. Precedence: defaults → `.env` → exported env var → CLI flag,
  which is `load_dotenv(override=False)`. Three knobs only:
  `RECOGNIZE_THRESHOLD`, `DETECT_CONF`, `DETECT_MAX_AREA`.
- `testdata/verify_scores.py` rewritten: it now crops through
  `robot.detect.padded_crop` **with the mask**, groups views by filename stem,
  prints the same/different margin and a threshold sweep, and takes `--source`
  so anyone can calibrate against their own camera. The old version could not
  have caught this bug — it validated a pipeline the app doesn't run.
- README gained "Calibrating For Your Camera".

`Robot.teach(crop, transcript)` is unchanged — see the dead ends below.

### Two dead ends, both closed

**A static crop-quality gate** (minimum size, Laplacian sharpness, mask-fill
fraction) was calibrated against ground-truth promiscuity across all 49 live
crops and predicts it poorly: the best signal was the crop's short side at
Spearman −0.74, sharpness −0.60, and mask-fill fraction was **noise at
+0.08**. The Mirror crop that broke the demo is 141x171 with Laplacian
variance 553 — neither small nor blurry, so a static gate passes it.

**A scene-relative teach-time gate was then built and cut.** It scored the
candidate crop against the other embedded tracks in view and refused above a
fraction of them. It was cut because it does not do the job it was built for:
it compared at `memory.threshold`, and at the new 0.90 the Mirror crop claims
only 6% of the room, well under any sane refusal bar. Its supporting number
(79% of the room) was measured at the **old 0.80 bar** — evidence gathered
before the fix that made it unnecessary. Raising the threshold is what
neutralized that crop, on its own.

The general lesson, if this tempts anyone again: a degenerate crop is only
degenerate *relative to the operating threshold*. Calibrate the threshold; do
not filter crops. A refusal path also costs a failure mode a learner cannot
diagnose, in a repo whose rule is the smallest change that works.

## UI responsiveness — stutter on the buttons, and a hopping UNKNOWN box

Two operator-visible symptoms, five causes: three behind the stutter, two behind
the hopping box. All five were found by reading the threading in `robot/app.py`,
not by profiling on the device; each is a structural cost that is obviously
there once named, but **the operator has not yet confirmed the feel on the
Jetson**. If a stutter remains, profile before changing anything else — the
cheap explanations are now used up.

The operator-facing half of this is in the README under "Aiming It" (how to move
a sticky focus) and "What to teach" (why you teach an object that is still in
frame). Keep those two in sync with the constants here.

Nothing here touches the crop pipeline, the encoders or the threshold. Score
parity re-run anyway and is byte-identical: same-object min 0.887 / med 0.920,
different med 0.472 / max 0.611, margin +0.275.

Headless replay over `testdata/` also prints the same verdicts as before, but
**do not read that as proof of no behaviour change**: those seven images hold
one object each, so they never exercise the focus stickiness below, and each
image gets its own `detector.reset()`. The replay test covers the crop and
recognition path, not the attention path. The attention path is covered by
reasoning and by throwaway stub tests, not by anything in the repo.

### Symptom 1: the feed stalls when you press FORGET or REBOOT

Three separate costs were sitting on the frame-pump thread, the one thread that
must never block — it does `cap.read()` (103 ms), composes the view and encodes
the JPEG the browser is streaming.

1. **Key handling ran inline in `_loop`.** `FORGET` scrolls up to 10 000 points
   with payloads, deletes, and `flush()`es; `REBOOT` closes and reloads the
   shard. Both also take `self.lock`, which the detect thread holds for an
   entire YOLO pass plus one CLIP embed per due track — so the wait alone could
   exceed half a second before the shard work even started. Now `_key_loop` /
   `_handle_key` run on their own thread; `_loop` only captures and renders, and
   takes no lock at all. One key thread, deliberately, so two taps on the same
   button run in order. Note this serializes the **key queue only** — phone
   hold-to-talk arrives on HTTP handler threads (`/listen`, `/audio`) and is
   still gated only by the non-atomic `self.busy` check-and-set, so a FORGET tap
   and a phone TEACH can still overlap. Pre-existing, and the shard writes are
   serialized by `self.lock` regardless.
2. **Teach and forget marked *every* track `last_query = 0`**, so the next
   detect pass fired a CLIP embed (~0.16 s) for each of N tracks in a single
   locked burst. Now each path requeries only what it must: teach stamps the one
   track it taught (stashed as `pending_track` when the crop is grabbed, beside
   the existing `pending_crop`), and forget stamps only the tracks wearing the
   deleted label. One embed instead of N.
3. **The panel re-read thumbnails off disk every rendered frame** — up to three
   `cv2.imread` calls at 10 fps. Now `_thumb_img`, an `lru_cache(32)`. Thumbs
   are write-once, so it cannot go stale (`Robot._thumb` stamps filenames with
   `time_ns`; `forget` deletes points, not files). Bounded on purpose: memory is
   the scarce resource on this board.

Also: `Track.requery_now()` sets `last_query = now - REQUERY_SECONDS` instead of
`0`. Zeroing it makes `last_query` falsy, which blanks the panel to "looking..."
for a beat right after a successful teach and reads as a glitch. And the
label-stripping on forget lives **inside `Robot.forget()`**, not in the app: it
is part of what forgetting an object means, and putting it there also catches
tracks that are momentarily off-screen and would otherwise keep a deleted name
for a couple of seconds. The box turns red on the press.

### Symptom 2: the UNKNOWN box jumps between objects, so you teach the wrong one

`Robot.focused()` was a stateless `max(t.salience)`, and salience is recomputed
every frame from box geometry. Two objects of similar size and centrality
therefore traded the box several times a second, and only one unknown is ever
displayed or teachable — so the teach target could change between deciding to
press and pressing.

- `focused()` is now sticky with `FOCUS_MARGIN = 1.25` in `robot/core.py`: the
  incumbent keeps focus until a challenger is 25% more salient or its track
  leaves the candidate list. It is an instance method now, holding
  `self._incumbent` — **keyed by the `Track` object, not by `tid`**. See the
  trap below; do not "simplify" it back to ids.
- **Both decisions are now made once per detect pass, inside
  `Robot.process_frame`**, and published as `Robot.attention` (the panel
  subject and FORGET target) and `Robot.teachable` (the salient unknown, the
  TEACH target). The app reads them. Previously the frame pump re-derived both
  every rendered frame, which meant the render thread was mutating the
  incumbency map with no lock while the detect thread mutated it too. Now there
  is a single writer, on the thread that owns track state, and only two pools
  ("view" and "unknown") instead of a third that never had more than one
  candidate.
- `Track.frames` now **decays** on a missed detection (`max(0, frames - 1)`,
  capped at `STABLE_FRAMES + 1` on the way up) instead of resetting to 0. A
  reset hid an established box for three more detect passes on a single
  detector blink, which was most of the visible flicker. Death is still
  time-based via `DEAD_SECONDS`.

The cap is deliberately one frame above the gate, not more. Every frame of
credit is a frame in which an object that has *left* still has a box and a
**stale crop** — and is still teachable. Whip an object out of frame and tap T
inside that window and you teach a picture of where it used to be, which is the
blank-crop poisoning described under "Recognition threshold". One blink of
tolerance buys the flicker fix; three would widen that window for nothing.

If the box still feels twitchy, `FOCUS_MARGIN` is the one number to turn up.
1.25 was picked by eye as a value comfortably above frame-to-frame jitter and
comfortably below a deliberate move; it was **not** measured against live
footage. Raising it makes focus harder to move on purpose, which is its own
failure.

### Two traps found in review, both closed — don't reopen them

Both were caught by an adversarial review of the diff, not by testing, and
neither would have shown up on the bench quickly.

- **Never key attention state by `tid`.** `Detector.reset()` calls the
  ultralytics tracker's `reset()`, which calls `reset_id()` and restarts track
  ids from 1 (`ultralytics/trackers/byte_tracker.py`). A tid-keyed incumbent
  therefore survives the reset and hands its focus to whatever *new,
  unrelated* track inherits its number — verified: a 0.50-salience newcomer
  held focus over a 0.60 one purely by inheriting an id. It bit `replay()` (a
  reset per image) and the live REBOOT beat. Holding the `Track` object makes
  the whole class impossible: a dead object simply isn't in the candidate list.
- **`robot.close()` must hold `self.lock`.** Once FORGET/REBOOT moved off the
  main thread, Ctrl-C during an in-flight forget could close the Edge shard
  from under a scroll/delete/reopen. Previously impossible, because those ran
  on the very thread that then did the shutdown. Shutdown now takes the lock.

### Rejected here

- **A dwell timer on focus** (challenger must lead for N ms) on top of the
  margin. The margin alone covers the jitter, and a second mechanism doing the
  same job costs a learner more than it buys.
- **Debouncing the buttons.** The stutter was the work on the wrong thread, not
  the press rate. Once `_handle_key` is off the pump, a double-tap is harmless:
  the second FORGET finds the label already cleared and says so.
- **Draining the key queue after a teach/ask.** `_process` used to call
  `_drain_keys()` to discard presses that piled up during the ~6 s
  transcription. With a dedicated key thread nothing piles up — presses are
  consumed within 0.2 s — so the drain was not just dead but dangerous: two
  consumers on one queue means `get_nowait()` after `empty()` can raise, and
  escaping that `finally` before `self.busy = False` wedges teach and ask until
  restart. The remaining `_drain_keys()` call in `run()` is fine: it runs before
  the key thread starts, so it is the only consumer.

## Measured baselines on this board

Useful reference numbers; re-measure before trusting them after any change.

- YOLOE-11L-seg-pf on CUDA at `imgsz=640`: **~165 ms/frame**, stable, 1.8 GB.
  Seconds per frame on CPU, so CUDA matters.
- Camera (Arducam 1080P on USB 2.0): negotiates **YUYV 1280x720 at 10 fps**,
  103 ms per `cap.read()`. This, not the GPU, caps the feed.
- Whisper-base via onnx-asr on CPU: load ~2.7 s, **transcribe ~6 s** for a short
  utterance. Thread count barely moves it (5.9 s at 6 threads vs 6.5 s at 2).
- Nomic load ~4 s, then ~0.15 s per embed. CLIP vision embed ~0.16 s.
- Live app: 2.3 GB / 18 threads at startup; 3.4 GB / 34 threads after one ask.
  Add one thread for the key handler (see "UI responsiveness").
- Score parity (must hold), from the rewritten `verify_scores.py`, which crops
  **with the mask** as the app does: same-object min 0.887 / median 0.920,
  different-object median 0.472 / max 0.611, margin +0.275. The older numbers
  (0.983 / 0.935 / 0.524) came from unmasked crops and are not comparable.
- Two featureless gray crops score 0.951 against each other — teaching a blank
  or background crop creates a memory that matches everything, which looks
  exactly like a recognition bug and is not one. Nothing rejects such a crop;
  a calibrated threshold is what keeps it harmless. See "Recognition
  threshold".

## Over-current (OC3) — understood, not a fault

`oc3_event_cnt` climbs continuously under load and OC1/OC2 stay at zero. This is
expected and is **not** a power budget being exceeded.

- OC3 counts *instantaneous* `VDD_IN` power spikes caught by a hardware
  comparator. NVIDIA staff: *"Instant high power may occur in very small time
  slot so that it cannot be captured by INA3221 sensor... They are detected by HW
  and we can only monitor the count instead of the value."* That is why
  `tegrastats` shows nothing near the limit while the counter rises.
- Measured here: sustained 1.9 A against a **4.05 A** critical limit, sampled
  peak 2.27 A, CPU pinned at max 1510 MHz, detector latency flat. No measurable
  cost to the demo. NVIDIA describes the throttle as protective, not damage.
- The popup comes from `/usr/share/nvpmodel_indicator/`, which polls the counters
  every second and re-fires while any of them increments. Suppressed on this box
  via `Hidden=true` in `~/.config/autostart/nvpmodel_indicator.desktop`; the
  applet was also killed for the current session. Hardware protection is
  untouched — `oc3_throt_en` is still `1`. Delete that file to restore the popup.
- Read the counters with:
  `cat /sys/devices/platform/soctherm-oc-event/hwmon/hwmon*/oc?_event_cnt`
- Reference: https://forums.developer.nvidia.com/t/oc3-raised-at-current-lower-than-critical/321989

## Settled questions — do not re-litigate

Each of these was investigated and closed. Reopen only with new evidence.

- **Native `sm_87` arch-list gate: abandoned, and it was the wrong gate.** The
  stock PyPI `torch 2.13.0+cu130` reports `['sm_80', 'sm_90', ...]` without
  `sm_87`, prints a scary compute-capability warning, and *works*: real CUDA
  matmul, and 165 ms/frame detection. Orin executes the `sm_80` kernels, which is
  what NVIDIA staff say. The warning is cosmetic. `torch`/`torchvision` are now
  pinned explicitly in `pyproject.toml` rather than arriving via Ultralytics.
- **NVIDIA PyTorch containers: unnecessary.** `nvcr.io/nvidia/pytorch:26.02-py3-igpu`
  does pass a strict native-`sm_87` gate, and a derived image ran the full app.
  It was dropped because the gate it satisfies does not matter and a container is
  heavy machinery for a beginners course. `Dockerfile.jetson`,
  `requirements.jetson.txt` and `.dockerignore` were deleted. Note that NVIDIA
  ends standalone iGPU images at release 26.03.
- **Jetson AI Lab SBSA wheels (`pypi.jetson-ai-lab.io/sbsa/cu130`): rejected.**
  The newest py3.12 aarch64 wheel reported only `['sm_110', 'sm_121']` and failed
  a real CUDA op with `no kernel image is available`. Targets Thor, not Orin.
- **Source-built PyTorch: never needed.**
- **DLA: unavailable on R39.2, and unnecessary.** `libnvdla_compiler.so` is
  absent from the BSP; NVIDIA confirms it and says a future TensorRT release will
  restore it. Do not symlink or copy libraries to fake it. The app does not use
  DLA. https://forums.developer.nvidia.com/t/missing-libnvdla-compiler-so-in-jetpack-7-2-on-jetson-orin-nx/372389
- **JetPack 6.2.1 / R36.4.4 downgrade: not needed.** It was only ever attractive
  for DLA, which the app does not use, and R36.4.4 has a documented DLA
  engine-build issue. A reflash would discard a working stack and force full
  revalidation on Ubuntu 22.04.
- **Subprocess worker for the speech/text encoders: replaced.** An earlier fix ran
  Whisper/Nomic/CLIP-text in a one-shot child process per interaction. Lazy
  loading achieves the same memory result in four lines, so
  `robot/interaction_worker.py` and `tools/` were deleted.
- **MJPG camera capture: tried and reverted.** It does raise the camera to 30 fps
  at 33 ms/read, but A/B measurement showed it **doubles the OC3 event rate**
  (+10 vs +5 per 40 s), costs ~46% more of a core and ~100 mA, for smoothness the
  detector cannot use at its 0.12 s gate.
- **Launch-time `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` tuning: not needed.**
  The in-code `ENCODER_THREADS` setting covers it properly.

## Operational notes

- The Jetson has no audio hardware. Use the phone-browser hold-to-talk path;
  `--host 0.0.0.0` serves HTTPS with a self-signed cert generated into `cert/`,
  which is what lets the phone grant mic access. Verified: page and MJPEG stream
  both return 200 over the LAN IP. `sounddevice` stays a project dependency for
  laptop use.
- **The cert warning cannot be removed, only made trustable.** No public CA will
  sign a private LAN address, and reaching one needs internet this demo is built
  not to need. What *was* wrong: the cert carried **no `subjectAltName` at all**
  (`-subj /CN=l6-robot` only), and browsers stopped reading CN in 2017 — so it
  named nothing, produced the harsher name-mismatch interstitial, and could not
  have been fixed by trusting it. It was also valid 10 years, which Safari
  rejects outright (>825 days) trusted or not. Now: SANs for the served IP plus
  `127.0.0.1`/`localhost`/`<host>.local`, 397 days, and `basicConstraints
  CA:TRUE` so a phone can install it as a root and stop warning for good —
  Android will not install a leaf without `CA:TRUE`, iOS will. Served at
  `/cert.crt` because a headless robot has no other easy way to hand a file to
  a phone; that gives away nothing the TLS handshake already does.
  Regenerated whenever the address changes, tracked by `cert/names.txt`, so
  trust is per-address: a phone trusted on home Wi-Fi warns again at the booth.
  Built with an openssl **config file**, not `-addext`, which the LibreSSL that
  ships as `/usr/bin/openssl` on macOS does not have.
  Verified end to end: with the cert loaded as a CA, hostname verification
  passes over a real TLS handshake against `StreamHandler` (so a phone that has
  installed it sees no warning), `/cert.crt` returns the bytes on the wire, an
  untrusted client still refuses it, and an unchanged address does **not**
  regenerate — a phone you trusted stays trusted across restarts.
- Close Firefox and the Codex/ChatGPT Electron app before demoing. They hold
  ~3 GB and are the only reason swap appeared in late measurements.
- `uv sync --locked --dry-run` wants to swap `nvidia-cusparselt-cu13 0.8.1`. It is
  a pre-existing lock quirk, harmless; `uv run --no-sync` avoids the churn.
- When testing with scripted teach/ask, always point `--data` at a scratch
  directory and check what the crop actually was. A teach against a blank panel
  poisons the shard in a way that mimics a threshold regression.

## If performance ever needs more

The demo is comfortable now, so none of this is scheduled. In rough order of
value, and each one needs a score-parity re-run because they change the crops:

1. **TensorRT FP16 export of the current YOLOE checkpoint.** Officially
   supported, likely the best memory/latency win, removes PyTorch from
   steady-state inference. Prompt-free export, masks, tracking and engine build
   on-device all need validating. Do not start with INT8.
2. **A smaller YOLOE segmentation checkpoint.** Less memory and compute, but it
   changes which objects and masks appear. COCO-pretrained closed-class
   alternatives (YOLO11n/s-seg, YOLOv8n/s-seg) only propose 80 categories and can
   miss arbitrary demo objects; FastSAM-s is the closest fit to the project's
   "find a thing, discard the label" philosophy.
3. **ByteTrack instead of BoT-SORT.** BoT-SORT defaults to sparse optical-flow
   camera-motion compensation, which costs CPU. Does not change embeddings, but
   changes track stability and IDs, so replay-test it.

References: https://docs.ultralytics.com/models/yoloe ·
https://docs.ultralytics.com/integrations/tensorrt ·
https://onnxruntime.ai/docs/performance/tune-performance/threading.html
