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

**Fourth milestone, same day: the box became a headless appliance.** It serves
its own Wi-Fi, starts the robot at boot, and needs no keyboard, screen, or venue
network. Everything is installed and running on this machine and was verified
live; the one thing **not** yet proven is a cold boot with the desktop gone,
which needs a reboot the operator has to do. See "Headless appliance" below.

**Fifth issue, same day: over the hotspot the feed ran slow and arrived 5+ s
late.** Two causes, both fixed and both measured on this box: 60% of the bytes
on the wire were the same frame sent again, and the hotspot was on 2.4 GHz.
See "Streaming over the hotspot" below. Frame rate confirmed good by the
operator afterwards.

**Sixth issue, same day: teaching by voice returned nonsense labels and each
action seemed to freeze for seconds.** Neither was headless and neither was the
robot — measured, no action stalls the frame pump by more than 149 ms. Whisper
was being handed the whole button hold, silence included, which both makes it
hallucinate and costs 16 s. Fixed by trimming to the speech region and by
keeping the browser mic open. See "The voice path" below. **Not yet confirmed by
the operator**, and one limit is unfixed: whisper-base mishears bare single
words, so the demo script should say "this is my laptop".

**Seventh milestone, same day: the panel became HTML.** The UI was one
composited JPEG — panel drawn with `cv2.putText` at a fixed 480 px and
`hstack`ed onto the feed — which made "mobile responsive" structurally
impossible. The stream now carries the annotated feed only and the page polls
`/state`. **Verified on this box** (endpoints, memory, threads, an A/B on
bandwidth); **not yet looked at by the operator on a phone.** See "The panel is
HTML now" below.

**Eighth issue, same day: recognized objects came and went on a static scene,
and teach seemed to freeze.** Neither was the threshold and neither was new.
The cause was `DEAD_SECONDS = 1.5` destroying a `Track` on a routine detector
dropout, so the object returned as a stranger. **A move to 0.88 was measured on
the live shard and refused** — it buys 2 recognitions and 5 mislabels, and the
score series that looked like a knife-edge belonged to the mirror, an UNKNOWN
that *should* not match. Fixed: `DEAD_SECONDS` 5.0, `FOCUS_MARGIN` 1.6, Nomic
loading outside the lock, and the status row no longer shifting the layout. Two
hazards found and left alone on purpose (a permanent person-blacklist, and lens
distortion). See "Objects came and went on a static scene" below.

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
- PR #3 (the responsiveness work under "UI responsiveness") is **merged** into
  `main` as of 2026-08-12, as is the cert work under "Operational notes". Note
  the code merged before the operator confirmed the feel on the device; that
  confirmation is still outstanding.
- PR #4 (the headless appliance described under "Headless appliance") is
  **merged** into `main` as of 2026-08-12. As with #3, it merged before the
  operator confirmed it on the device: **a cold boot with the desktop gone has
  still not been proven.**
- PRs #6 (streaming + voice) and #7 (the HTML panel) are **merged** into `main`
  as of 2026-08-12. Both merged before the operator saw the panel on a phone;
  that is still unconfirmed.
- Branch: `codex/track-lifetime-and-focus`, one commit ahead of `main`: the work
  under "Objects came and went on a static scene" — `DEAD_SECONDS`,
  `FOCUS_MARGIN`, `warm_encoders`, and the status row. No new dependencies. It
  also **refuses** a threshold move to 0.88 on measurement, which is the part to
  read before anyone tries it again.
- Branch: `codex/remote-access-doc`, one commit ahead of `main`: `REMOTE-ACCESS.md`
  and two pointers to it, no code. Documentation only — how to get a shell on the
  headless unit over any of three independent routes, and how to move it between
  Wi-Fi networks. The one machine change it records is that `tmux` was installed.
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

> **That re-measure has since been done, and 0.88 was refused.** On the live
> shard it buys 2 recognitions and 5 mislabels, and no taught view gains a
> single one of its own sightings, because the misses sit at 0.75–0.81 rather
> than just under the bar. Read "Objects came and went on a static scene" before
> acting on the paragraph above — the sentence "0.88 is the next stop" is what
> nearly caused a regression, and the thing that saved it was checking whether
> the near-bar scores belonged to an object that *should* match. They did not:
> they were a mirror.

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

- `focused()` is now sticky with `FOCUS_MARGIN` in `robot/core.py` (**1.6 now,
  raised from the 1.25 this section shipped** — see "Objects came and went"): the
  incumbent keeps focus until a challenger is that much more salient or its track
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
  time-based via `DEAD_SECONDS` — **now 5.0 s, not 1.5**; see "Objects came and
  went on a static scene" below. Note which constant does what: this decay is
  the *box's* lifetime, `DEAD_SECONDS` only decides whether the object comes
  back as itself.

The cap is deliberately one frame above the gate, not more. Every frame of
credit is a frame in which an object that has *left* still has a box and a
**stale crop** — and is still teachable. Whip an object out of frame and tap T
inside that window and you teach a picture of where it used to be, which is the
blank-crop poisoning described under "Recognition threshold". One blink of
tolerance buys the flicker fix; three would widen that window for nothing.

If the box still feels twitchy, `FOCUS_MARGIN` is the one number to turn up.
1.25 was picked by eye and **was not enough** — it shipped at 1.25 and was later
raised to 1.6 on an operator report plus a measured baseline of 3.5 focus
switches per minute; see "Objects came and went on a static scene". Raising it
makes focus harder to move on purpose, which is its own failure, and **nobody
has measured a deliberate takeover at 1.6.** That is the thing to watch.

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
  margin. A second mechanism doing the same job costs a learner more than it
  buys. **The claim in this bullet that "the margin alone covers the jitter" was
  wrong at 1.25** — it did not, and the operator said so. The timer is still
  rejected, on a better argument now; see "Objects came and went".
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

## Headless appliance — boots into the demo, serves its own Wi-Fi

The unit goes in a case with no keyboard, screen, or peripherals, and demos in
rooms with no usable network. So: **apply power, and it boots into the robot on
its own access point.** Operator-facing instructions are in the README under
"Headless Appliance", and getting a shell on it afterwards — three independent
routes plus a serial console for when none of them answer — is `REMOTE-ACCESS.md`.
Keep the SSID and password quoted in those two files in sync with
`deploy/headless-setup.sh`, which is where they are defined, and the address in
sync with both the script and the unit (see the knobs paragraph below). The band
is 5 GHz for a measured reason — see "Streaming over the hotspot".

`deploy/` holds the two tracked artefacts: `l6-robot.service` and the idempotent
`headless-setup.sh` that installs everything. Re-run the script after editing
either. It is safe to re-run while the robot is live: it will not bounce an
access point that is already serving, and rather than restart the robot under
someone's hands it prints the `systemctl restart l6-robot` needed to pick up a
changed unit. The Wi-Fi block is deliberately **last**, after every step that
needs internet — the review list below records what the wrong order did.

Only `SSID` and `PSK` are knobs (`sudo SSID=my-robot ./deploy/headless-setup.sh`).
The address is deliberately not one: `10.42.0.1` is fixed in both the script and
the unit's `--advertise`, because a hotspot on one address and a certificate
naming another is precisely the name-mismatch warning this whole path exists to
avoid — an `AP_IP` knob existed briefly and was cut for exactly that reason. The
service account is read out of the unit's `User=` rather than being a second
knob that can disagree with it.

### The five changes to the machine

Service enabled, hotspot profile with every saved network parked, desktop off
(`multi-user.target`, which also retires the OC3 popup applet), sshd, model
cache under `$HOME`. The README tables all five with their undo commands under
"What The Setup Changes" — keep that table authoritative rather than mirroring
it here. One fact specific to this box: **ssh host keys were missing**, so sshd
was enabled but dead; `ssh-keygen -A` creates only what is absent.

### Two bugs this found, both fixed in the app

**FastEmbed was caching 1.1 GB of encoders in `/tmp`.** Its default is
`/tmp/fastembed_cache`, and `systemd-tmpfiles` prunes `/tmp` on a 30-day rule
(`/usr/lib/tmpfiles.d/tmp.conf`, `D /tmp 1777 root root 30d`). A robot on its
own hotspot has no internet, so the first boot after that sweep would hang in a
download that can never finish — a demo that dies of old age, weeks later, for
no visible reason. `robot/models.py` now passes `cache_dir=CACHE_DIR`
(`~/.cache/fastembed`, honouring FastEmbed's own `FASTEMBED_CACHE_PATH`), and
the existing cache was moved there rather than re-downloaded. Whisper was
already safe: onnx-asr uses `~/.cache/huggingface`.

Verified genuinely offline, not just with an offline flag: all four encoders
load in ~2.5 s each inside `sudo unshare -n` (a namespace with nothing but
loopback). So **no `HF_HUB_OFFLINE=1` is needed** in the unit — considered and
rejected, because setting it would break a fresh install's first download for
no benefit.

`testdata/verify_scores.py` needed the same fix and was missed at first: it
builds its own `ImageEmbedding` rather than going through `robot.models`, so
running the calibration script re-downloaded 336 MB into `/tmp` — a second copy,
in the directory the change exists to get out of. It now passes `CACHE_DIR` too.
If you add another entry point that embeds anything, pass it as well; that is the
whole reason `CACHE_DIR` is a module constant and not a default buried in a call.

**A second SIGINT was aborting the shutdown, leaving the shard open.** systemd's
default `KillMode=control-group` signals *every* process in the cgroup, so `uv`
forwarded one SIGINT and the kernel delivered another. The second landed inside
`server.shutdown()`, the `KeyboardInterrupt` escaped past `cap.release()` and
`robot.close()`, and the process died in a native destructor: `terminate called
without an active exception`, exit 134. Observed, not theorised. The shutdown
sequence now lives in a `finally:` that starts with
`signal.signal(SIGINT, SIG_IGN)`, which also fixes a human holding Ctrl-C — the
same bug by hand. `KillMode=mixed` would have fixed only the systemd half, so it
was not used; the app is now correct however it is launched. After the fix: stop
takes 5 s, `Deactivated successfully`, exit 0, no traceback.

Worth knowing: the shard was **not** corrupted by any of those aborts (61 points,
reopens clean). Every write is flushed at upsert time, which is what makes
pulling the plug the documented off switch.

### Two new flags, both for unattended running

- **`--advertise ADDR`.** `run()` derived the URL and certificate address from
  `_lan_ip()` whenever `--host` was non-loopback, which on a hotspot with no
  default route returns `127.0.0.1` — a certificate naming the wrong address,
  which is the harsher name-mismatch interstitial and cannot be fixed by
  trusting it. The service passes `--host 0.0.0.0 --advertise 10.42.0.1`: listen
  everywhere (so Ethernet stays a maintenance path), name the hotspot address.
  Because that address never changes, **phone trust is now permanent** — the
  per-address regeneration warned about under "Operational notes" no longer
  bites, since there is only ever one address. Verified: `cert/names.txt` is
  `IP:10.42.0.1,IP:127.0.0.1,DNS:localhost,DNS:qdrant-robot.local`. It takes an
  **IP only**, enforced in argparse: `_ensure_cert` writes it as an `IP:` SAN, a
  hostname there makes openssl reject the certificate, and the `except` around
  the TLS setup then serves plain http — the phone mic disappears over a typo,
  with one line of warning. `loopback` is derived from `host`, never from the
  advertised address; deriving it from `addr` silently downgrades
  `--host 0.0.0.0` to http on a box with no default route.
- **`--watchdog SECONDS`** (service uses 30, default 0 = off). A USB camera that
  fails cleanly already ends `_loop`, because `cap.read()` returns False. This is
  for one that wedges *inside* the driver call, which the frame pump cannot
  notice because it is the thread that is stuck. It `os._exit(1)`s — deliberately
  not a clean close, since `self.lock` may never come free, and safe because
  writes are already flushed. Default off because on a laptop a closed lid is
  indistinguishable from a stalled camera.

  Two details it is easy to get wrong, and this code got wrong first:

  - **It arms its own clock (`frame_at = monotonic()`) instead of waiting for a
    first frame.** A `frame_at is None` guard looks like it protects warm-up, but
    the thread only starts *after* `warm_up()` and `detector.warm()` return, so
    the guard's only real effect was to exempt a camera that wedges on its very
    first read — which is exactly what a power-cut USB device does, and exactly
    what the flag exists for. The robot would sit on the splash screen forever,
    with no restart and no log line, on a box with no screen.
  - **The clock is `time.monotonic()`, not `time.time()`.** The operator is told
    to set the appliance's date by hand while it runs (see "The clock" below), and
    a wall clock stepped a day forward reads as a stall and kills a healthy robot.
    A backward step is worse: it blinds the watchdog for the length of the offset.

  Covered by a throwaway stub test, not by anything in the repo: fires on a stall,
  fires when no frame ever arrives, quiet while frames arrive, exits promptly on
  `self.stop` (it waits on the event rather than sleeping, so a shutdown cannot be
  shot from under itself). Monotonic immunity is by construction, not tested — you
  cannot step the clock on a live box to prove it.

`PYTHONUNBUFFERED=1` is in the unit for a duller reason: Python block-buffers
stdout into a pipe, so `journalctl -u l6-robot -f` showed only torch's startup
warning while every robot log line sat in a buffer — and on the abort above,
that buffer was lost.

### Verified live, and the one thing that is not

Verified on this box, service running as `qdrant` under systemd: page and
`/cert.crt` return 200 over HTTPS on `https://10.42.0.1:8765`, the MJPEG stream
delivers ~32 MB in 5 s, RSS **2.33 GB / 19 threads** — the documented healthy
startup baseline, so CUDA and the encoders load fine with no session and no
display. The AP is real (`iw dev`: `type AP`, channel 6, 20 MHz), with dnsmasq
and a `nm-shared-wlP1p1s0` MASQUERADE rule, which is also why the hotspot
carries internet whenever Ethernet is plugged in.

**Not verified: a cold boot.** `set-default multi-user.target` takes effect on
the next reboot, and nobody has rebooted since. If it comes up wrong, sshd is
the way back in — over the hotspot, or Ethernet.

### Found by adversarial review of the diff — don't reintroduce these

None were found by testing; the appliance looked fine on the bench with all of
them present, which is the point.

- **The watchdog's `frame_at is None` guard, and its wall clock** — both
  dissected under "--watchdog" above.
- **`ssid` set only on profile create, not on modify.** Re-running with a new
  `SSID=` — the advertised use — changed the password and printed the new name
  while the radio kept broadcasting the old one.
- **Wi-Fi ran first, cutting the radio before the 1.1 GB encoder download that
  needed it** — and cutting your own ssh session if it came in over Wi-Fi. The
  Wi-Fi block is last now, and must stay last.
- **An `AP_IP` knob let the script and the unit's `--advertise` disagree**,
  reintroducing the certificate mismatch. Fixed by deleting the knob, not by
  syncing machinery: the address is fixed in both files.
- **The parking loop split `nmcli -t` output on `:`**, so a profile name
  containing a colon stayed free to grab the radio at boot. It iterates by UUID.
- **The README's older manual-hotspot recipe still lacked `--advertise`**, so
  following it reproduced the mismatched certificate. Fixing the service path
  is not fixing the docs path.

### The clock, an open wart with two documented fixes

Not a bug in the code, but it shows in the demo: recall speaks timestamps, and
an offline robot's clock is whatever `systemd-timesyncd` last saved to
`/var/lib/systemd/timesync/clock` — there is no NTP on its own hotspot, and the
board keeps no time across a power cut without a coin cell on the RTC backup
battery connector (`J3` in the carrier board spec; `J13` beside it is the fan).
So a unit switched on cold at a booth stamps memories with the time it was
packed away. README "The Clock, Which Recall Reads Out Loud" gives the two
fixes: `timedatectl set-time` over ssh, or fit the cell. Neither is done here —
the box is on Ethernet and NTP-synced right now, which hides it.

### Rejected here

- **A physical power button.** Unnecessary for *on*: NVIDIA documents the kit as
  powering on automatically when DC is applied. The carrier board does have a
  button header, but the pin numbers live in the Carrier Board Specification PDF
  behind NVIDIA's download login, and the same header carries force-recovery and
  reset — so the README says confirm against the board and does **not** print
  pin numbers. Do not fill them in from a blog post.
- **A user-level service with lingering.** A system unit with `User=qdrant` needs
  no login session to exist, which is the whole point of a headless box.
- **Automatic fallback between venue Wi-Fi and the hotspot.** Rejected: it needs
  a dispatcher script, and the address then depends on which branch won, which
  puts the certificate warning back on every phone. A fixed address is worth
  more than automatic internet.

## Streaming over the hotspot — slow feed, 5+ s behind the room

Symptom, reported from a laptop joined to the robot's own Wi-Fi: low frame rate
and a delay over five seconds. It had been fine on the desk over Ethernet, which
is the tell — **nothing about running headless is slow; the radio was.**

Two causes, both measured here rather than reasoned about.

**60% of the bytes were duplicate frames.** The `/stream` handler pushed
`self.jpeg` on a fixed `time.sleep(0.04)` tick — 25 fps — while the pump renders
at the camera's 10 fps. Measured over 7 s on loopback: 171 frames sent, **69
unique**, 32.2 MB, **37 Mbps**. Every third-of-a-frame the browser decoded was a
frame it already had. Free on Ethernet, which is why it survived this long.

The fix is a sequence number, not a timer. `LiveApp._publish` stamps each
composed JPEG and stores `self.shot = (seq, bytes)` — **one tuple, assigned in
one bytecode**, so the number and the bytes cannot disagree the way two separate
attributes could. The handler sends only when the seq changes, and always reads
the *latest* shot, so a client that falls behind loses frames instead of
accumulating lag. After: 68 frames, 68 unique, **13.1 Mbps at 9.7 fps** — the
same frame rate, a third of the bytes. `self.jpeg` is gone; `splash()` publishes
through the same method.

**The hotspot was on 2.4 GHz channel 6 at 20 MHz.** That carries 15–25 Mbps in
an empty room and much less in a hall full of phones, against a 13 Mbps demand —
so the link was the narrowest part of the chain. The failure mode is what makes
it confusing: TCP does not drop what it cannot send, it queues, so the feed stays
smooth and falls further behind the longer you watch. It reads as a slow robot.
Now **5 GHz channel 44**, set in `deploy/headless-setup.sh` (`BAND`/`CHANNEL`,
named once, not knobs). Verified live: `type AP`, `channel 44 (5220 MHz)`, same
SSID, same `10.42.0.1`, dnsmasq and the MASQUERADE rule intact, page 200.

Channel 44 rather than the alternatives: 36, 48 and everything from 52 to 140 are
`no IR` in this box's world regulatory domain, so an AP cannot initiate there —
`iw phy phy0 info` leaves exactly 40, 44, 149 and 157. Of those, 44 is non-DFS
and clear of the 149–165 block venue kit crowds into. NM 1.46 has no AP channel
width property, so it is HT20; 5 GHz HT20 on an empty band is already several
times the demand, and forcing HT40 was not needed.

**Because the address never changes, none of this touched certificate trust** —
that is the whole point of the fixed `10.42.0.1` and `--advertise`.

### Two things this is not

- **Not a headless penalty.** RSS and thread count are unchanged from the
  documented baseline; there is no desktop to blame and CUDA is fine.
- **Not the camera or the detector.** 9.7 fps *is* the camera's ceiling (103 ms
  per `cap.read()`). The stream now delivers every frame the pump composes, so
  there is nothing left to win on the network path without lowering quality.

### The knob, if a venue is worse than this

`STREAM_QUALITY` in `robot/app.py`, was a bare `85` at the `imencode` call.
Measured on a real composed 1760x720 frame: 183 KB at 85, 154 at 75, 134 at 70,
100 at 60 — i.e. 15.0 / 12.6 / 11.0 / 8.2 Mbps at 10 fps. Left at **85**: the
duplicate fix and the band move together give enough headroom that paying in
image quality was not justified. Turn it down only against a measurement.

Diagnose in this order, because it separates the robot from the radio:
`timeout 5 curl -sk https://127.0.0.1:8765/stream -o /dev/null -w '%{size_download}'`
measures what the app *wants* to push with the radio out of the picture; if that
is comfortable and the phone still lags, it is the link. `iw dev wlP1p1s0 info`
says which band the AP is actually on, which is not always what a profile says —
see below.

### Found by review, closed

- **The script's "don't bounce a live AP" branch would have lied.** It exists so
  a re-run cannot drop connected phones, but a changed band or channel only
  reaches the air on a bounce, so it would have printed the new channel while
  the radio kept broadcasting the old one — the same silent-mismatch trap as the
  `ssid`-on-modify bug above, and it would have sent the next person debugging
  the wrong layer. It now compares `iw dev` against `CHANNEL` and prints the
  `nmcli con up l6-hotspot` needed. Confirmed by watching the channel half fire
  on this run. It compares the **name as well as the channel**: a re-run with a
  new `SSID=` is the advertised use, and that is the case that would send a room
  to a network that does not exist.
- **`BAND`/`CHANNEL` are named once.** Hardcoding 44 in both the `con modify`
  and the check is the `AP_IP` mistake again: two places free to disagree.
- **The README's manual hotspot recipe still said plain
  `nmcli device wifi hotspot`**, which defaults to 2.4 GHz — so following the
  docs path reproduced the bug the service path had just fixed. It now passes
  `band a channel 44`. Fixing the service is not fixing the docs, which is the
  second time that exact trap has come up in this repo.

## The voice path — hallucinated labels, and an action that looks like a freeze

Reported together with the streaming problem above, and asked as "why is
headless worse?". **On the evidence, headless is not worse and the robot is not
freezing.** Measured on loopback with the radio out of the picture, sampling
every frame gap for 14 s while firing an action at the 4 s mark:

| | frames | median gap | max gap | stalls > 400 ms |
|---|---|---|---|---|
| idle | 136 | 101 ms | 122 ms | none |
| during FORGET | 136 | 102 ms | 125 ms | none |
| during a full voice ask | 136 | 102 ms | 149 ms | none |

So the "UI responsiveness" work above is confirmed on the device by measurement,
at least on the server side: nothing the buttons do reaches the frame pump. The
harness is `gaps.py` in the session scratchpad — throwaway, not in the repo.

What is real is that **an action takes 16 seconds**, and the feed running
perfectly while nothing else happens reads exactly like a freeze.

### The cause: Whisper was being handed the button hold, not the utterance

The actual failing capture was still on disk at `/tmp/l6-utterance.wav` — 48 kHz
mono, **7.72 s long, holding 1.4 s of speech**, the rest room noise at rms 4–20.
Transcribed as-is and then trimmed to the speech region:

| | duration | transcribe | result |
|---|---|---|---|
| as uploaded | 7.72 s | **16.3 s** | `'L L L L L L L L L…'` |
| trimmed to speech | 2.00 s | **5.2 s** | `'La-caw.'` |

That is the operator's exact symptom reproduced from their own audio. Whisper
hallucinates repetitions when given long silence, and it charges for every
second it is given — so one generous hold caused both the nonsense label and the
16 s wait. `is_silent` does not catch it: the clip's overall rms is 929, well
over the 120 floor, because a quarter of it *is* speech.

`audio.trim_to_speech(path)` now cuts both ends in place before `transcribe`,
using the same `SPEECH_RMS` bar `record_wav` stops on, with 0.2 s of pad. It is
idempotent, returns the kept duration for the log line, and no-ops on a WAV that
is not 16-bit mono or where nothing crosses the bar.

**It fails open, and there is a gap between the two bars.** `is_silent` passes
anything at rms 120 or above, while the trim needs a 100 ms block at 200 — so a
soft-spoken hold is neither rejected nor trimmed, and gets the full-length
treatment with the hallucination that comes with it. Left as is on purpose:
locating speech in a clip that quiet means guessing, and the rms line in the log
already tells the operator to speak up. Do not close it by lowering
`SPEECH_RMS` — `record_wav` stops recording on that same number, and a lower bar
means the laptop path stops on room noise.

**The word itself is a separate, unfixed limit.** `La-caw` was "laptop", and it
stays `La-caw` after trimming. Gain is not the reason — the clip already peaks at
41% of full scale, and normalizing to 50% and 90% returned `La-caw` both times,
so **level normalization was tried and rejected on measurement**. It is
whisper-base on a bare isolated noun. The fix is operator-facing and free: say
"this is my laptop", which is also the phrase `parse_label` is written to read.
README "What to say" carries this.

### The browser mic was rebuilt on every press

`start()` ran `getUserMedia`, built an `AudioContext`, and compiled the worklet
*after* the finger was down, and `stop()` tore all three down again — so every
utterance paid the setup cost, during which nothing is captured, and the page did
that work on the same main thread that renders the MJPEG feed. Now opened once
by `mic()` and kept for the life of the tab, warmed on the first `pointerdown`
anywhere so the first teach of a demo is not the one that misses its own word.

Note what this is **not** evidence of: the 2.9 s of quiet at the front of that
7.7 s clip is *captured* room noise, not a late start, so it does not measure the
setup latency. The reason to keep the mic warm is structural, and the cost of
being wrong about it is one clipped word.

Two details worth keeping:

- **`mic()` memoizes the promise, not `ctx`.** `ctx` is only assigned after three
  awaits, so an `if (ctx) return` guard lets the warm-up listener and the first
  press both through, and two worklet nodes pushing into one chunk list is the
  audio recorded twice over. On failure it clears the memo so a retry is
  possible.
- **The `AudioContext` asks for 16 kHz.** That is Whisper's rate, so the browser
  resamples instead of sending 3x the bytes *up* the same half-duplex Wi-Fi link
  the feed is coming *down* — 740 KB became ~250 KB for that clip. Browsers that
  ignore the option report their real rate in `ctx.sampleRate`, which is what the
  WAV header is written from, so it stays honest either way. (Per spec an
  unsupported rate *throws* rather than falling back, so there is no silent
  wrong-rate path to worry about — only a mic that fails to open.)
- **A kept mic needs the recovery the per-press version got for free.** Opening
  it once means nothing re-opens it when it dies: a backgrounded tab suspends the
  `AudioContext` (an incoming call does it on iOS), and the track ends outright
  if permission is pulled or the input device changes. Either one records silence
  behind a healthy-looking recording indicator until someone reloads — mid-demo,
  on a phone. So `start()` resumes a non-running context inside the press (the
  one place `resume()` is allowed), and `onended` clears the memo so the next
  press opens fresh. `openMic()` closes the outgoing context on the way, or a
  demo that loses the mic a few times hits the browser's live-context limit.

The mic staying open means a permanent recording indicator in the browser. That
is the price, and it is in the README.

## The panel is HTML now — the UI was one composited JPEG

"Ugly, unreadable, not mobile responsive" were one structural fact: **the panel
was pixels drawn into the video**, fixed at 480 px and `hstack`ed onto the feed.
On a phone in portrait the browser contains a 2.44:1 image into ~160 pt of
height, so the panel rendered about **100 pt wide** — which no font-size tuning
fixes, and which is why the page used to carry a "rotate to landscape" nag.

`/stream` now carries the annotated feed alone; the page polls `/state` (JSON,
4 Hz) and renders the panel as DOM. Gone: `draw_panel`, `draw_gauge`,
`draw_match`, `_thumb_img`, `_wrap`. `draw_feed`/`_chip` stay — boxes and their
labels do belong in the video. Two supporting endpoints: `/thumb?f=NAME`
(`Path(name).name`'d, so nothing outside the thumbs dir is reachable however
the query is written) and `/crop.jpg` (the live "sees now" crop, encoded per
request, 404 when nothing holds focus).

Measured here: **~13% fewer bytes**, A/B on one scene, 49 frames in 5 s either
way, 241 KB composed vs 210 KB feed-only. Do not sell this as a bandwidth fix —
a flat panel of text is cheap to JPEG, and the camera view was always the
expensive part. RSS/threads unmoved at 2.47 GB / 19 under sustained polling.

### Decisions — don't reopen

- **REBOOT lost its button; the `R` key stays.** The operator asked why it
  existed, believing it wiped memories. It does the opposite, but a control
  named after its mechanism taught nobody that, and a robot on its own Wi-Fi
  with no internet already makes the point. The wipe is `--reset`, still
  CLI-only. Four buttons fit a phone; five did not, so this also removed the
  need for an overflow tray.
- **No device split.** "Desktop only" was rejected for REBOOT: the appliance is
  headless, so the phone is the only UI it has.
- **Polling, not SSE.** `protocol_version` stays HTTP/1.0, so each poll is its
  own connection — measured as costing nothing. Moving to 1.1 needs an accurate
  `Content-Length` on *every* response or clients hang.
- **Never build the panel with `innerHTML`** — labels and notes are whatever
  Whisper heard. The page goes through a small `el()` helper using
  `textContent`.
- **Browser speech (`speechSynthesis`) considered, not done.** It would fix the
  Jetson answering silently, but it double-speaks on a macOS host driving its
  own tab, and iOS wants a gesture the ~16 s round trip has outlived.
  `_answer_line` is split out of `_speak` so the panel at least shows the
  sentence.

### Found by adversarial review, closed

- **The feed could not shrink, so a landscape phone got no panel at all.**
  `#view` was `flex:0 0 auto` with no cap, and `#side` is `flex:1` — basis 0,
  so it contributes nothing to the used height and cannot push the feed back. A
  full-width 16:9 image is exactly the height of a 667x375 viewport, so the
  panel computed to **zero**: precisely the failure this change exists to fix,
  reintroduced by dropping `flex:1; min-height:0` from `#view`. Fixed with
  `max-height:50vh` in the column layout (reset to `none` in the row layout, or
  it clamps the feed there too), and the row breakpoint lowered to 640 px so
  landscape phones get the side-by-side layout at all.
- **`banner` is a property so that assigning it bumps a counter.** The page
  hides a status line after 6 s, and keyed on the *text* it could not tell a
  stale message from the same message sent again — press IGNORE twice a minute
  apart and the second press was silent. `status_seq` is the fix; don't
  "simplify" the property away. Verified: three identical FORGET refusals
  produce three distinct seq values.
- **`/state` sends `Cache-Control: no-store`.** Nothing else stops a client
  caching it, and a frozen panel beside a moving video is a horrible thing to
  diagnose in a room.
- **`PAGE` is a bytes literal, so it is ASCII-only, comments included.** An em
  dash in a CSS comment is a `SyntaxError`, not a rendering bug. There is a note
  above the literal.

### Not verified

**Nobody has seen it on a phone.** Firefox is installed but headless screenshots
do not run under snap confinement here, so the layout is checked by endpoint
tests and by construction, not by eye. Judge the portrait split and whether the
type reads across a room; `#label` is `clamp(20px,4.5vw,30px)`.

## Objects came and went on a static scene — and the bar was not the problem

Reported together with "teach freezes into *thinking* for a few seconds". Two
symptoms, and **the threshold change they seemed to call for was measured and
refused.** Read that part before touching `RECOGNIZE_THRESHOLD` again.

### The bar stays at 0.90. This was measured, not argued.

The tempting move was 0.90 → 0.88, which CLAUDE.md had already pre-authorised
("0.88 is the next stop"). It is wrong here, and the shard says so. Every
sighting scored against every taught view, on the live shard (809 sightings,
8 taught views):

| bar | recognized | correct label | mislabelled | unrecognized |
|---|---|---|---|---|
| 0.90 | 768 | 724 | **44** | 41 |
| 0.88 | 775 | 726 | **49** | 34 |
| 0.86 | 785 | 728 | **57** | 24 |

0.88 buys 2 more correct recognitions and 5 more mislabels. And per taught view
the number of its own sightings missed is **identical** at both bars — 124/124,
126/126, 74/74, 42/42, 78/78, 14/14, 0/0 — because the misses sit at 0.75–0.81,
nowhere near the bar. Nothing is being dropped *just* under 0.90.

The evidence that looked like a knife-edge was a **series belonging to an
UNKNOWN** — the mirror, reading 0.86/0.87/0.89/0.90 on consecutive requeries.
That is a degenerate reflective crop scoring against a taught object, i.e. the
promiscuity the 0.90 calibration exists to prevent, and lowering the bar would
have let it start claiming labels. Taught objects in the same frames were at
0.92–0.98 throughout. The harness is `bar.py` in the session scratchpad
(throwaway); it loads a **copy** of the shard so the live service is untouched,
which is the way to re-run this.

Lesson worth keeping: a score series near the bar means nothing until you know
whether it belongs to an object that *should* match. Check that first.

### What actually caused it: `DEAD_SECONDS`, and it was the wrong 1.5 s

A real object's detector confidence swings either side of `DETECT_CONF`. Measured
on the operator's own frames, the microphone's own proposal came back as
`'camera'`/`'microphone'` at **0.34–0.91** against a floor of 0.30, and was
absent altogether on some frames. Dropouts past a second are routine, so at
`DEAD_SECONDS = 1.5` the `Track` was destroyed — taking `label`, `note`,
`thumb`, `vec` and `last_query` with it — and the object returned as a stranger:
nothing on screen for `STABLE_FRAMES` passes, then a fresh embed, then green
again. **That is the "comes and goes", and the shard counts it:** 866 sighting
thumbs, and in one 25-minute window 57 `Track` births across only 17 tracker
ids, ids repeating (4315 twelve times, 5187 nine times). A repeating tracker id
means the *tracker* never lost the object. Only our `Track` did.

Now 5.0 s. BoT-SORT's own `track_buffer` is 30 frames — and a "frame" here is a
detect pass, so 9–15 s of wall clock at the measured 0.1–0.6 s pass rate. 1.5 s
was far stricter than the tracker underneath it. Safe because the box's
lifetime is the `frames` decay, not this: two missed passes take a track under
`STABLE_FRAMES`, and `Detector.process` returns only stable tracks — so a
lingering track is not displayed, not embedded, not teachable, and **the
stale-crop teaching window is not widened.** Side benefit: far fewer `sighted`
resets, so the shard stops accumulating a thumbnail and a flushed upsert per
rebirth.

Residual risk, bounded: within `track_buffer` BoT-SORT can re-attach a buffered
id to a different object that drifts into the old position, which now arrives
wearing a stale label until its next requery (≤2 s). Not tid reuse — ids are
monotonic within a run, and `reset()` clears `self.tracks` anyway. Reasoned, not
observed.

Measured after: **Track births 4.12/min → 2.33/min** (99 births in the 24 min
before the restart, 43 in the 18.4 min after, excluding the first minute's
startup burst, which necessarily reboots every track). Treat that as ~40% and no
better: the windows are unequal and the scene was not controlled — a person was
working at the desk throughout. An earlier reading of 1.80/min came from a
7-minute window and did **not** hold up as the window grew; the honest figure is
the one above.

And be clear about what did **not** change: **box disappearances off `/stream`
were 68 in 130 s before and 68 in 130 s after.** That is expected — the box's
lifetime is the `frames` decay, not this constant — but it means the *visible
blink* the operator sees is not reduced by this round. What is fixed is that the
object comes back as itself, immediately, instead of as an unnamed stranger.
Anyone who wants the blink itself gone has to make detection steadier (a lower
`DETECT_CONF` is the free knob to try first) — not touch `DEAD_SECONDS` again.

### The rest of the round

- **`MAX_DET` is not the limit, and nothing is saturating.** The operator asked
  whether undisplayed objects are still detected and whether a cap is being hit.
  They are still detected *and CLIP-embedded* — only knowns plus one unknown are
  ever drawn — but YOLOE emits **9–10 proposals per frame against
  `MAX_DET = 64`**. Roughly 8 of ~10 stable tracks are embedded and discarded
  every 2 s, which is real waste, and it explains **nothing the operator sees**:
  detect passes measure 0.1–0.6 s (requery gaps 2.1–2.6 s against a 2.0 s
  cadence), three times clear of even the old `DEAD_SECONDS`. A
  `MAX_EMBEDS_PER_PASS` cap was designed against this and **cut** — the same
  build-then-delete shape as the crop-quality and scene-relative gates. Do not
  rebuild it without a measurement showing passes are long.
- **`FOCUS_MARGIN` 1.25 → 1.6**, and the dwell timer stays rejected. Measured
  off `/crop.jpg`: **3.5 focus switches/min before (median dwell 2.34 s), 0 in
  120 s after** — on one scene, by a thumbnail-fingerprint heuristic that cannot
  tell "fixed" from "now too sticky to move on purpose", which is the failure to
  watch for. A timer was reconsidered on the operator's request and refused: any
  hold **gated on candidacy** is powerless against a blink, since `focused()`
  can only retain an incumbent still in the candidate list and that list is
  stable tracks only. A hold that returned a non-candidate *would* work and is
  worse — it aims the panel and TEACH at an object with no box and a stale crop.
  **Be clear about what this does not fix:** the blink half of the churn
  survives, and a bigger margin makes a returning object work harder to reclaim
  its place. `DEAD_SECONDS` does not help here either — it preserves identity on
  return, it does not stop the handoff. What softens it in practice is an
  accident: `focused()` returns None on an empty candidate list without clearing
  the incumbent, so an object that blinks while nothing else is teachable
  resumes focus when it returns.
- **Nomic loads outside the lock.** `models.warm_encoders(kind)` is called in
  `_process` after `transcribe` and before `with self.lock:`. Both loaders are
  `lru_cache`d, so it is the same work relocated. It matters because Nomic's
  first load is ~4 s and paying it under the lock stalls the detect thread long
  enough to trip `DEAD_SECONDS` — the app causing the flicker it is trying not
  to have. `Robot.teach`'s signature is unchanged, deliberately.
- **`#status` no longer toggles `display`.** It reserves a 38 px row always and
  toggles `visibility`, so the button bar stops jumping under a waiting finger
  on every action. `line-height` is pinned so the row cannot lose to a UA's font
  metrics by a pixel, and `nowrap` keeps a long transcript from wrapping to two
  lines and reintroducing the shift. A regression from the HTML panel commit.

### Teach was never slower — the panel just started admitting it

From the operator's own journal: **6 s steady, 10–11 s on the first teach of a
process.** That is whisper-base on this board plus the one-time Nomic load, and
it is what the numbers under "Measured baselines" already predict. Neither of
the last two commits slowed it; `trim_to_speech` made it faster. What changed is
that `da92caa` made "thinking..." *legible* — the same text used to be drawn
~100 px wide into the video. "It used to not do that" is true of the sign, not
the duration. Do not go looking for a regression here again; if the wait is the
problem, the fix is the wording or the first-teach load, not transcription.

### Two hazards found and deliberately left alone

Both are real, both were reported to the operator, and neither is fixed:

- **`is_person_like` blacklists a track id permanently.** `_person_tids` is
  never cleared for the session, and `PERSON_WORDS` matches by word, so one
  frame classified person-ish removes an object for the whole demo. Confirmed
  live: a `person` box sits over the microphone at **0.77–0.91 conf on nearly
  every frame** while the mic's own proposal flickers near the floor, and one
  proposal in the scene comes back as `'baby bottle'` — `baby` is in the list,
  so that object is permanently un-teachable. `hair dryer`, `head phones`,
  `arm chair`, `eye glasses` go the same way. Needs a design decision (decay the
  flag? require two consecutive hits?), not a one-liner.
- **The lens has visible barrel distortion.** A crop's geometry therefore
  depends on where in frame it sits, so an object taught near the centre and
  seen near the edge is a different shape to CLIP — part of the score spread
  above. It also biases `salience`, which already weights centrality.
  Undistorting is `cv2.calibrateCamera` + one `initUndistortRectifyMap` and a
  ~2–4 ms `remap` per frame, but it **invalidates every existing memory**, since
  all of them were taught through the distorted lens, so it forces a re-teach
  and a score-parity re-run. A deliberate project, not a flicker fix. Free
  mitigation meanwhile: teach and demo near the centre of frame.

## Measured baselines on this board

Useful reference numbers; re-measure before trusting them after any change.

- YOLOE-11L-seg-pf on CUDA at `imgsz=640`: **~165 ms/frame**, stable, 1.8 GB.
  Seconds per frame on CPU, so CUDA matters.
- Camera (Arducam 1080P on USB 2.0): negotiates **YUYV 1280x720 at 10 fps**,
  103 ms per `cap.read()`. This, not the GPU, caps the feed.
- Whisper-base via onnx-asr on CPU: load ~2.7 s, **transcribe ~6 s** for a short
  utterance. Thread count barely moves it (5.9 s at 6 threads vs 6.5 s at 2).
  Cost scales with clip length, and that dominates everything else here: 2.0 s of
  audio costs 5.2 s, 7.7 s of audio costs **16.3 s**. Trimming silence is worth
  more than any thread tuning — see "The voice path".
- Nomic load ~4 s, then ~0.15 s per embed. CLIP vision embed ~0.16 s.
- Live app: 2.3 GB / 18 threads at startup; 3.4 GB / 34 threads after one ask.
  Add one thread for the key handler (see "UI responsiveness").
- Same app as a systemd service, no desktop session: **2.3–2.5 GB / 19 threads**,
  i.e. no penalty for running headless. A clean `systemctl stop` takes 3–5 s, all
  of it the shard close. The desktop it replaces was holding ~1.5 GB.
- MJPEG stream: the frame is 1280x720 since the panel moved to HTML. **Scene
  dominates size**, so re-measure rather than quoting: 210 KB at quality 85 on a
  cluttered desk (**~17 Mbps** at the camera's 10 fps), against 183 KB recorded
  earlier for the 1760x720 composed frame on a tidier scene. Same-scene A/B of
  the change itself: 241 KB composed vs 210 KB feed-only, 49 frames in 5 s
  either way — **~13%**. Was 37 Mbps before the duplicate-frame fix. The hotspot
  (5 GHz, HT20) carries it; 2.4 GHz did not. See "Streaming over the hotspot".
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
  **The appliance sidesteps that**: it serves its own hotspot at a fixed address
  and names it with `--advertise`, so there is only ever one address to trust and
  trusting it is permanent. See "Headless appliance".
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
