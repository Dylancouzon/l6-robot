# Jetson Deployment Status

Durable handoff for the NVIDIA Jetson deployment. Update it when a test changes a
conclusion or a milestone completes.

**Status at 2026-08-12: solved, shipped, and operator-confirmed.** The robot
runs on the Jetson from a plain `uv` install — no container, no reflash, no
source-built PyTorch — as a headless appliance on its own 5 GHz hotspot,
driven from a phone. Thirteen rounds of work landed the same day; each has a
section below with the measurements. The cold boot the earlier rounds flagged
as unproven has since happened: the box booted at 13:13 into
`multi-user.target` and the service was serving 14 seconds later. The operator
teaches by phone, so the HTML panel and the voice path are confirmed in daily
use.

The thirteen rounds, newest last — each heading below carries the full story:

1. **Startup memory** (not CUDA) made the view update every 15 s — lazy
   encoder loading fixed it. See "The problem and the fix".
2. **Recognition matched everything** — the 0.80 threshold was calibrated
   against crops the robot never produces; now 0.90. See "Recognition
   threshold".
3. **UI stutter and a hopping UNKNOWN box** — work moved off the frame pump,
   focus made sticky. See "UI responsiveness".
4. **Headless appliance** — own Wi-Fi, boots into the demo. See "Headless
   appliance".
5. **Slow, lagging feed over the hotspot** — duplicate frames and 2.4 GHz,
   both fixed and measured. See "Streaming over the hotspot".
6. **Hallucinated labels, actions that read as freezes** — Whisper was handed
   the whole button hold; now trimmed to speech. One limit unfixed:
   whisper-base mishears bare single words, so say "this is my laptop". See
   "The voice path".
7. **The panel became HTML** — the old composited-JPEG panel could not fit a
   phone. See "The panel is HTML now".
8. **Objects came and went on a static scene** — `DEAD_SECONDS` was killing
   tracks on routine dropouts. A threshold move to 0.88 was measured and
   REFUSED; read that section before touching `RECOGNIZE_THRESHOLD`. See
   "Objects came and went on a static scene".
9. **Recognition-quality and teach-latency review** — labels lowercased,
   model loads hidden under the button hold, multi-view teaching measured as
   the big recognition win, swap re-enabled, live shard cleaned. See
   "Recognition quality review" below.
10. **FORGET deleted the wrong object** — it was aimed by focus, which moves
    under a finger already on its way down. There is a MEMORY tab now: every
    taught object with a picture, its taught views, and the delete. See "The
    memory tab" below.
11. **People stopped being filtered out** — the person blacklist is deleted, so
    "this is Dylan" works like any other teach; the tab also gained RENAME and
    DROP THIS VIEW. See "People are objects now".
12. **"The system slowed down a lot"** — two findings, one fix each way: the
    slowdown itself was a laptop routing its internet through the hotspot
    (documentation, not code), and the review it prompted found the memory
    tab waiting 0.8–1.6 s behind YOLO for half-millisecond reads (code). The
    tab also gained SET LOCATION. See "The slowdown, the lock, and the
    location button".
13. **"Where did I leave my hat?" answered "dylan"** — recall was searching
    the image space with the words of the question, and that ranking is
    measured noise. Recall now picks the object in the text space and
    answers with its last three sightings by time; the SEEN/HEARD split and
    CLIP's text tower are gone. See "Recall answered from the wrong space".
14. **Three phone reports** — the mic permission prompt on every use (browser
    behaviour, README fix, no code change), ignores dying with the session
    (they persist as vectors in the shard now and re-suppress by appearance),
    and the memory tab not showing the robot's own photos (the card's
    tap-cycle now walks taught views, then recent sightings). See "The
    prompt, the clutter that came back, and the photos the tab couldn't
    show".

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
- PRs #1–#10 (Jetson CUDA, threshold calibration, UI responsiveness + certs,
  headless appliance, streaming + voice, HTML panel, track lifetime + focus,
  memory tab + people as objects, recall from the text space + SET LOCATION)
  are all **merged** into `main` as of 2026-08-13. Their rationale lives in
  the sections below — including #8's **refusal** of a threshold move to
  0.88 on measurement, which is the part to read before anyone tries it
  again.
- Branch: `codex/persistent-ignore-and-sightings` (PR #11), off `main`:
  round 14 — "The prompt, the clutter that came back, and the photos the
  tab couldn't show" below. No new dependencies; one new payload kind
  (`ignored`), additive to existing shards. Deployed and serving on the
  unit since 2026-08-13.
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
  loop uses per frame. The others load lazily on first use via `lru_cache`
  (four encoders then; three since CLIP's text tower was deleted — see
  "Recall answered from the wrong space"),
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

Do not treat swap as the solution: a correct configuration does not touch it.
It IS the OOM backstop, though — the swapfile was later found deactivated
entirely and was re-enabled; see "Recognition quality review".

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
not by profiling on the device. Since confirmed by measurement (see "The voice
path": no action stalls the frame pump by more than 149 ms) and by daily use.
If a stutter ever returns, profile before changing anything — the cheap
explanations are used up.

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
   `pending_teach`, which was `pending_crop` until the memory tab gave a teach a
   frame and a box to stash too), and forget stamps only the tracks wearing the
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

### Verified live, cold boot included

Verified on this box, service running as `qdrant` under systemd: page and
`/cert.crt` return 200 over HTTPS on `https://10.42.0.1:8765`, the MJPEG stream
delivers ~32 MB in 5 s, RSS **2.33 GB / 19 threads** — the documented healthy
startup baseline, so CUDA and the encoders load fine with no session and no
display. The AP is real (`iw dev`: `type AP`, channel 6, 20 MHz), with dnsmasq
and a `nm-shared-wlP1p1s0` MASQUERADE rule, which is also why the hotspot
carries internet whenever Ethernet is plugged in.

**The cold boot has since been proven** (2026-08-12): the box booted at 13:13
into `multi-user.target` with no desktop and the service was serving 14
seconds later, per the boot journal. Nothing about this milestone remains
unverified.

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
  need for an overflow tray. **Still four**: the memory tab did not add a fifth,
  it took FORGET's slot — see "The memory tab".
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

### Since confirmed on a phone

The operator now drives the demo from a phone daily — teaches and asks go
through the panel's hold-to-talk — so the layout works in practice. The knob if
type ever reads too small across a room: `#label` is `clamp(20px,4.5vw,30px)`.

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
- **Nomic loads outside the lock.** `models.warm_encoders` (then taking a
  `kind` argument, since removed) is called in
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
the duration. Do not go looking for a regression here again; the first-teach
load has since been moved under the button hold (see "Recognition quality
review"), and the steady 6 s is transcription's floor on this CPU.

### Two hazards found and deliberately left alone

Both were real and both were reported to the operator. **The first is now
gone** — not fixed, deleted, see "People are objects now" below:

- ~~**`is_person_like` blacklists a track id permanently.**~~ `_person_tids` was
  never cleared for the session, and `PERSON_WORDS` matched by word, so one
  frame classified person-ish removed an object for the whole demo. Confirmed
  live at the time: a `person` box sat over the microphone at **0.77–0.91 conf
  on nearly every frame** while the mic's own proposal flickered near the floor,
  and one proposal in the scene came back as `'baby bottle'` — `baby` was in the
  list, so that object was permanently un-teachable. `hair dryer`,
  `head phones`, `arm chair`, `eye glasses` went the same way. The whole
  mechanism was removed when people became teachable, which takes the hazard
  with it. Kept here because it explains anything odd in a shard taught before
  that.
- **The lens has visible barrel distortion.** A crop's geometry therefore
  depends on where in frame it sits, so an object taught near the centre and
  seen near the edge is a different shape to CLIP — part of the score spread
  above. It also biases `salience`, which already weights centrality.
  Undistorting is `cv2.calibrateCamera` + one `initUndistortRectifyMap` and a
  ~2–4 ms `remap` per frame, but it **invalidates every existing memory**, since
  all of them were taught through the distorted lens, so it forces a re-teach
  and a score-parity re-run. A deliberate project, not a flicker fix. Free
  mitigation meanwhile: teach and demo near the centre of frame.

## Recognition quality review — teach latency, labels, views, swap

A full review pass (2026-08-12, ninth round) on "teaching still takes some time
to register, recognition works okay but not great". Everything here was
measured on the live box or the live shard before it was changed.

### Teach latency: 6 s is the floor, and it is Whisper

From the journal, release→"taught" is a flat 6 s whether the trimmed speech was
2.2 s or 1.4 s. That is whisper-base on CPU plus ~0.3 s of embeds and the shard
write. The GPU option was checked and closed: this venv's onnxruntime 1.27.0 is
the CPU-only build (`get_available_providers()` = CPU/Azure), so GPU Whisper
means sourcing a Jetson ORT-GPU wheel — heavy machinery against a pinned stack,
for one demo beat. **Do not chase the 6 s.** Threads buy 0.5 s at best (already
measured), and a smaller Whisper mishears more.

What WAS cut: the first voice action of a session paid ~7 s of model loads
(Whisper ~2.7 s + Nomic ~4 s, measured 7.1 s together) *after* the finger
lifted. `on_listen` (the phone path) now starts `models.warm_encoders`
on a daemon thread at pointerdown, so those loads run under the button hold.
Trap that makes the lock necessary: **`lru_cache` does not serialize a cache
miss**, so the pointerdown warm and `_process` calling the same loaders
concurrently would build the same model twice — double the load time and a
transient extra model's worth of RAM. Hence `_warm_lock` in `robot/models.py`,
and `_process` calls `warm_encoders` BEFORE `transcribe` so a fast release
waits on the lock instead of racing it. Verified: a second caller entering
0.5 s into a warm finishes with the first, wall time unchanged (7.1 s), third
call instant.

**Be honest about what "under the hold" buys — the review measured it.** A
cold ONNX model load holds the GIL in 1.4–1.6 s blocks, and every thread —
frame pump, detect, `/state` — pauses with it. A ticker thread on a 20 ms
cadence lost 75–89% of its wakeups during a cold `warm_encoders("t")`; a warm
call costs nothing (0.1 ms). So the first press of a session freezes the feed
for ~3 s in two blocks while "LISTENING" is up. **Relocated, not removed**:
the identical stall previously ran after release behind "thinking...". Tracks
survive because the hold is ~3 s against `DEAD_SECONDS = 5.0` — that margin
is the load-bearing part, and it is not large. (The 149 ms max-gap number
under "The voice path" is a warm-session figure; it never covered a cold
start.)

**The laptop path deliberately does NOT warm at press.** The first version
did, and the adversarial review caught why it must not: `record_wav` re-enters
`stream.read` every 100 ms on the same interpreter, a 1.6 s GIL hold overflows
PortAudio's buffer, and `stream.read`'s overflow flag was being discarded —
so the first laptop teach of a session would record chopped audio and fail
like a Whisper hallucination, silently. The warm was removed there (the phone
records on the phone, so the phone path has nothing to starve), and
`record_wav` now prints a line when the overflow flag is set, so a starved
recording can never again masquerade as a transcription bug.

### Labels are lowercased at parse_label — the only door they enter through

The live shard held the same laptop as `laptop` (4 taught views) and `Laptop`
(2), and the same bottle as `Water bottle` and `water bottle`, because Whisper
capitalizes on a whim. Labels are exact-match keys: `forget` deletes by string
and recall dedupes by string, so each casing was a separate object — FORGET
left the twin behind, recall listed both. `parse_label` now lowercases, which
covers both mic paths at the single point every label passes through.

Writes alone don't close it, though — the review called this out. A learner
with a shard written before this change still holds `"Water bottle"`, and a
re-teach now writes `"water bottle"`: FORGET would delete only the
matching-case half and the box stays green, which reads as a broken delete.
So the three read-side comparisons are case-folded too — `Memory.forget`,
`_dedupe_by_label`, and the label-stripping loop in `Robot.forget` — which
fixes old shards with no migration.

The live shard was migrated the same day (service stopped, backup at
`~/edge-data.bak-20260812`): 213 labels case-folded, and the mirror — 530 of
976 points and the source of 26 of the 44 historical mislabels — forgotten
outright. After: 446 points, five labels, all lowercase. Trap for next time:
**`UpdateOperation.upsert_points` with an existing id does NOT rewrite the
point** in this qdrant_edge build — it reports nothing and changes nothing.
`UpdateOperation.set_payload(ids, payload)` is what works.

### Multi-view teaching is the recognition fix, and it needs no code

Scored on the live shard (938 sightings vs 14 taught views at the 0.90 bar):
a label's misses are a coverage problem, not a threshold problem. The laptop's
best single view covered 103 of its 173 sightings; the union of its 6 views
covered **all 173**. The mouse: 135/150 for the best view, 150/150 for its 2.
Re-teaching already adds views and recognition already takes the nearest, so
the entire fix is the operator habit — **teach each object 2–3 times, turning
it between teaches** — now in the README under "What to teach".

The remaining mislabels (44/938) were dominated by the mirror (26, now
forgotten) and by masked laptop *fragments* matching the mouse (17). No
margin/vote/second-opinion mechanism was added for that residue: it is the
same build-then-delete shape as the gates this file already buried, and the
data fix (views + don't teach reflectives) is what actually moves the number.

### Swap was off, and 4.2 GB RSS with no swap is an OOM waiting for a demo

The 8 GB `/swapfile` existed but was active in no sense: not in `/proc/swaps`,
not in fstab. Meanwhile the service sits at 4.2 GB RSS after a day of
interactions with ~2 GB available. `Restart=always` turns an OOM kill into a
30 s mid-demo restart. Re-enabled with `swapon /swapfile` plus an fstab line.
The rule from round one stands — a correct configuration does not *use* swap —
this is the backstop that turns a memory spike into slowness instead of a dead
demo.

### Open options, offered and deliberately not taken this round

- ~~**Person-blacklist decay**~~ — overtaken: the blacklist is gone entirely,
  see "People are objects now".
- **`DETECT_CONF` 0.30 → 0.25** in `.env` against the visible blink — zero
  code, revert if clutter grows.

## The memory tab — deleting stopped being something you aim

Reported as "the forget button doesn't work well: sometimes we click as the
focus goes on something else". That is not a bug to fix in FORGET; it is
FORGET's design. It acted on `Robot.attention` at the instant of the press,
and attention moves — a press is a decision plus a hand travelling, and the
box can change owner in between. `FOCUS_MARGIN` already went 1.25 → 1.6 to
slow that down (see "Objects came and went") and it cannot fix this: any
stickiness that made deletes safe would make teaching impossible to aim.

So deleting moved off the live view entirely. **`M` / MEMORY opens a
full-screen list of everything taught, and you delete the card you are
looking at.** No new dependencies, and no migration: this is additive to the
payload.

### What it is

- `Memory.objects()` — taught points, grouped by label, case-folded exactly as
  `forget` and `_dedupe_by_label` are, so the tab's idea of one object is the
  delete button's idea of one object. Newest first, with `count_label` giving
  the sighting count per label (5 labels = 6 ms, measured).
- Four endpoints: `GET /memories?offset&limit`, `GET /ignored`,
  `POST /forget?label=`, `POST /unignore?tid=` (now `?pid=`, a point id —
  round 14 made dismissals shard points). (Six now — `POST /rename` and
  `POST /forget_view` arrived in the next round, below.) The two mutations run
  synchronously in the HTTP handler under `self.lock` — an HTTP handler thread
  is already off the frame pump, which was the whole reason for the key
  thread, so routing them through the key queue would have bought nothing and
  cost the tab an honest answer to redraw from.
- **A "scene" picture per taught view** (payload key `scene`), because the
  crop CLIP compares is a poor thing to identify an object by — masked,
  gray-filled, often a fragment. Its framing changed once the same evening,
  from the whole frame to the object plus a margin; see "Two follow-ups"
  below for what it is now. JPEG quality 80 throughout, which was worth
  measuring: 43 KB against 89 KB at cv2's default 95, for a picture nothing
  scores against. Points written before it exists fall back to the crop, which
  is why an old memory looks gray beside a new one and why **nothing had to be
  deleted**, despite the operator offering.
- `Detector._ignored` went from a set of tids to `tid -> {tid, thumb, ts}`,
  so the tab can list what IGNORE dismissed and undo it. Nothing is stored for
  an ignored object, so un-ignoring is just removing the block.
- `Robot.ignore` now takes the **track and frame** rather than a tid, for the
  same reason: "IGNORED x3" with no pictures is not something anyone can undo
  with confidence.

### Decisions

- **MEMORY took FORGET's button; it is not a fifth.** Four still fit a phone.
  The `F` key stays — it is the fastest delete to film, and a key is not
  something a thumb lands on by accident.
- **A full-screen sheet, not a third column.** A phone has no width to spare,
  and browsing memory is something you stop to do, not something to watch
  beside the feed. It also drops the video (`img.removeAttribute('src')`,
  with `paused` guarding the reconnect handler) — on the hotspot the feed is
  most of the radio and none of it is visible behind the sheet.
- **Two-tap delete, armed then filled.** A list you scroll with a thumb is a
  list you tap by accident, and this delete is unrecoverable. No `confirm()`:
  it blocks the page and reads as a browser artifact on a phone.
- **`repeat(auto-fill, minmax(170px, 1fr))`, no breakpoint.** One full-width
  card per row put 2.4 objects on a phone screen, which is not a list you can
  scan; two columns puts 4.5 and still shows the picture at 173x97.
- ~~**Delete by object, not by view.**~~ **Overturned in the very next round,
  on the operator's request** — DROP THIS VIEW exists, see "People are objects
  now". The argument here was that teaching again is the fix for a bad view;
  it is not, because teaching again *adds* a view and leaves the bad one in
  memory matching things it shouldn't. Kept as written because it is a fair
  record of how thin the reasoning was: "a second delete path is machinery"
  was a guess about cost, and the cost turned out to be about fifteen lines.
- **Grouping scans every taught point per page request.** Eight points on the
  live shard. If a shard ever holds hundreds of objects, page the scroll
  itself rather than slicing the grouped list.

### How this was tested, and how to test it again

Every endpoint and both mutations were driven over HTTP with the service
stopped and the app pointed at a **copy** of the live shard (the rule under
"Operational notes"). A real teach goes through without a mic: `GET /listen?k=t`
then `POST /audio?k=t` with a WAV file.

The page itself needs a browser, and this is the part worth rebuilding rather
than re-deriving. The harness was a small proxy that serves the real `PAGE`
with the memory tab already opened and a probe script appended, and forwards
everything else to the app; headless Firefox `--screenshot` renders it, and
the probe POSTs its measurements back to the proxy, because there is no other
way to get numbers out. Two traps in it: firefox screenshots on the `load`
event, so hold that open with a slow-loading image or you photograph a
half-laid-out page; and the probe must be driven against a **stub server on a
scratch shard**, never the live service — see the keydown bug below for what
one wrong guess would have cost.

### Worth knowing

- ~~**A Q press writes a thumbnail nothing will ever reference again.**~~
  Overtaken in round 14: a dismissal is a shard point now, and the point
  references its thumbnail. What remains true is that TRACK AGAIN deletes
  the point and not the JPEG (the same policy as forget), so an *undone*
  ignore orphans its picture — 5–15 KB on an NVMe, still not a leak worth
  machinery.
- **The scene picture contains whatever was in frame, including people.** And
  since the round below, a person can be the *subject* rather than the
  background. Every scene captured during this work has the operator in
  it. That is fine for a desk demo and worth a thought before a shard travels.
- **No automated test covers any of this**, in keeping with the rest of the
  repo: the headless replay path never renders a page. The probe above is the
  only thing that has exercised the client, and it ran once.

## People are objects now, and the tab can edit as well as delete

Same day, on the operator's request: *"remove the human block — I actually
want this project to recognize different people"*, plus the two follow-ups
offered at the end of the memory-tab round (rename, drop one view). All three
landed together.

### The person filter is gone, not softened

`PERSON_WORDS`, `is_person_like` and `_person_tids` are deleted, along with the
`>4096` pruning that kept the blacklist bounded. `self.names` went with them,
and so did the `cls` column in the detect loop's `zip`: **nothing in the app
reads a detector class name any more.** That is worth noticing — the repo's
whole claim is "detection finds a thing, memory decides which thing", and the
one place that quietly consulted the detector's vocabulary was this filter.

Measured before touching it, on the live camera at `DETECT_CONF = 0.30`, so
the decision was not guessed:

| proposal | conf | area | band | salience |
|---|---|---|---|---|
| person (also read as `sculptor`) | 0.92–0.94 | **0.072–0.078** | KEPT | 0.19–0.20 |
| door | 0.71–0.75 | 0.089 | KEPT | 0.28 |
| typewriter / office desk / shelve (phantoms) | 0.32–0.59 | 0.21–0.36 | DROPPED | — |

So a seated person at desk distance sits **comfortably inside the 0.20 area
cap** and needs no calibration change — which was the open question, since the
cap exists to drop torso-sized boxes. A face filling the frame does exceed it;
back off or raise `DETECT_MAX_AREA`. The `.env` defaults are untouched.

What it costs, honestly:

- **A person often will not hold focus.** Confirmed live: with the operator
  standing in frame, the salient unknown was the *door* (0.28 against 0.20).
  Only one unknown is displayed and teachable, so teaching a person is the
  same aiming problem as any object — come closer, be central. Nothing here
  needs fixing; it is the documented behaviour of `salience`.
- **CLIP is not a face recognizer.** 512 dimensions from a 224 px crop keys on
  the whole silhouette, clothing included. Two people dressed alike are a
  harder separation than two objects, and a jacket change probably needs a
  re-teach. **Untested — one person was available.** If someone reports people
  being confused for each other, that is the thing to measure first, and
  multi-view teaching is the first answer, not the threshold.
- Hands and faces are now teachable clutter in principle. Not observed in the
  measured frames (the person box covered the operator), so treat it as
  unmeasured rather than absent. IGNORE dismisses them, and the memory tab can
  now undo that.

The privacy line is worth stating once, since this is teaching material: the
robot stores a vector and a photograph of whoever is taught, in a shard that
travels to demos. Teaching a person is a thing done with them, in front of
them, by saying their name out loud.

### Rename, and dropping one view

Both live on the object's card in the memory tab.

- **`Memory.rename`** rewrites the label on every point wearing it —
  `set_payload`, never a re-upsert (an existing id does NOT rewrite the point
  in this build; that trap is already recorded above). `Robot.rename`
  lowercases and collapses whitespace exactly as `parse_label` does, so a
  typed name and a spoken one normalize to the same key, and it updates live
  tracks **in place** rather than requerying: the vectors did not move, so the
  box just starts saying the new name. Verified live — 91 points renamed, and
  the box read `dylan couzon` at 0.973 with no re-embed. Renaming onto an
  existing label MERGES the two objects, which is deliberate: it is the cure
  for `Laptop` and `laptop` having been separate things.
- **`Robot.forget_view`** deletes one taught point. If it is the object's last
  taught view it forgets the whole object instead — otherwise the label's
  sightings outlive it and recall describes an object the robot can no longer
  recognize. The button only appears when there are 2+ views, so a single-view
  card offers FORGET and nothing more confusing.

### Two bugs the browser probe caught, both invisible to the server tests

Neither would have been found by curling the endpoints, and both are the kind
that ship:

- **Grid rows were being squeezed, so every card had its buttons clipped off
  with nothing to scroll to.** `#memlist` has a definite height (`flex:1` in a
  column), and with `auto` rows the browser fits the rows INTO that height:
  measured `rows=132px ...` against a 266 px card, `scrollHeight` equal to
  `clientHeight`. `align-content:start` does **not** prevent it — there is no
  free space to distribute, the space is negative. `grid-auto-rows:max-content`
  is the fix (132 → 284 px, scrollHeight 782 → 1346). It had looked fine at
  four objects purely because the row height the browser picked happened to
  exceed the content, which is why the earlier round's screenshots passed.
- **Point ids do not survive JSON in a browser.** They come from
  `itertools.count(time.time_ns())`, so ~1.8e18, well past JavaScript's 2^53
  safe integer — the page rounded the id, asked to delete one that does not
  exist, and the view **quietly survived its own deletion**. The server-side
  test passed because Python parsed the same JSON exactly. `_view_json` now
  emits the id as a **string**; the server parses it back. Anything else this
  page ever sends an id for needs the same treatment.

Two smaller ones from the same session: `.del` and `.drop` lost to
`.obj button` on specificity (0,1,0 against 0,1,1), so FORGET rendered violet
and DROP never got its own row — every modifier in that block now repeats the
`.obj button` prefix. And `/forget_view` must be routed **before** `/forget`,
because `startswith` matches both.

### Verified

Rename and both delete paths were exercised against a copy of the live shard,
including the cases that only a second tab produces (an id already dropped, an
id belonging to another object) and five malformed requests. Replay over
`testdata/` prints the same verdicts and `verify_scores.py` is byte-identical
to the baseline under "Measured baselines" — **that parity is the check to
re-run after anything in this area**, since nothing here should be able to
move it.

### Two follow-ups from watching the operator use it

Both reported the same evening, after he taught himself twice:

- **The tab's picture is now the object, not the room.** `_scene` stored the
  whole frame with a box drawn on it, which at 173 px on a phone card makes
  the object a detail you squint at. It now crops to the detector's box plus
  `SCENE_PAD = 0.3` of margin — wider than the recognition crop's `PAD = 0.12`,
  which is tuned for what CLIP should see, where this is tuned for what a
  person needs at thumbnail size and a little real background helps. No box is
  drawn any more: the picture *is* the box. Unmasked, unlike the embedded
  crop, because here the background is the point. 480 px longest side, and the
  files got smaller with it — 43 KB whole-frame against 5–15 KB cropped.
  Falls back to the whole frame when there is no box.

  The card CSS had to change with it: `object-fit:contain` at `4/3`, not
  `cover` at `16/9`. These are tight crops of arbitrary shape — a standing
  person is twice as tall as wide — and `cover` fills the box by cutting the
  ends off the very object the picture exists to show. Verified in a browser
  against generated crops shaped tall, wide, square and tiny: every one is
  fully visible, letterboxed on the axis it doesn't fill.

- **One box per remembered object** (`Robot._one_per_label`). Teach yourself
  twice — the recommended habit, and the biggest recognition win there is —
  and every taught view is a memory the detector's *other* boxes can match:
  the person box, the face box and the torso box all come back "dylan", so one
  memory is drawn three times over one human. Best match wins: highest score,
  then salience. Score only moves on a requery (every `REQUERY_SECONDS`), so
  the winner is stable between them rather than swapping frame to frame the
  way a salience-only rule would — measured over six frames of salience
  jitter, the box never moved, and it moved exactly when a requery flipped the
  scores.

  The losers keep their labels and their tracking; they are simply not drawn,
  not attended to, and **not written as sightings**, which also stops one
  object logging a sighting per box. Case-folded like every other label
  comparison here.

  **The accepted cost, which the operator named himself:** two identical
  objects in frame — two of the same mug — now show one box between them. The
  labels are the same string, so there is nothing to tell them apart with, and
  drawing one name twice was the confusing half of it.

  **The track holding attention keeps its box, whatever the scores say** —
  and that exception is the whole reason this is an instance method reading
  `self._incumbent`. Caught by review, not by the tests above, and it is a
  real trap: dropping a track out of `display` takes it out of `focused`'s
  candidate list, and `focused` can only retain an incumbent that is still a
  candidate. So a score flip between two boxes of one object bypassed
  `FOCUS_MARGIN` *entirely*, and attention did not move to the new winner —
  it re-derived by raw salience over what was left, which can be a **third
  object**. Reproduced with stubs: the panel jumped from "dylan" to a mug and
  back on the requery cadence, with the margin powerless. Anything else that
  ever filters `display` inherits this; the general rule is that a track
  holding focus must not be removed from the candidate list by a filter.

  Verified through the real `process_frame` with a stub detector: two "dylan"
  tracks plus a "mug" draw two boxes, one sighting is written rather than two,
  the loser keeps its label, two *different* objects are both still drawn, an
  unknown alongside a duplicated known is still the teach target, attention
  holds steady across five score flips (it jumped on every one before the
  fix), the survivor is promoted when the holder's track dies, and the
  incumbent does **not** freeze a worse match in place forever — lose
  attention and the best score wins the box again.

### Found by adversarial review of the diff — don't reintroduce these

An independent reviewer read the whole diff afterwards. Everything below was
already written, tested and believed correct; none of it was found by the
tests above, which is the same lesson this file has recorded three times now.

- **The rename field made every keystroke a robot command.** The page's global
  `keydown` listener fires `/key?k=` for `tarfq` and had no target guard, and
  the new `<input>` bubbles into it. Typing a name is then a string of
  commands: "water bottle" sends `a`, `t`, `t`, `r` — two voice actions and a
  **REBOOT of the shard** — `f` deletes whatever the camera is looking at, and
  `m` closes the sheet, which blurs the field and *commits the half-typed
  name*. Nearly every English label triggers something. Fixed by bailing on
  `INPUT`/`TEXTAREA`/`contenteditable` targets; verified in a browser against a
  stub server (typing `"water bottle from marta"` now produces only the
  background `/state` polls, and `f` outside a field still works). **Any new
  text field on this page inherits this trap** — the listener is global.
  Deliberately tested against a scratch stub and never the live service: if the
  guard had been wrong, the test itself would have rebooted the operator's
  shard and deleted whatever was in frame.
- **`openMem()` racing an in-flight `loadPage()` hid the first page.** Every
  delete and rename redraws by calling `openMem`, so a page request is very
  often in flight. `openMem` cleared the list and reset `memAt = 0`, but its
  own `loadPage()` no-opped on `memBusy`, and then the stale response inserted
  its old-offset cards into the freshly cleared list and set `memAt` to *its*
  offset — so objects 1–12 were unreachable until the tab was closed and
  reopened. Fixed with a generation counter (`memGen`), plus clearing
  `memBusy` in `openMem` so the fresh page can start immediately; the stale
  response checks the counter before touching the DOM *or* the flag.
  Reproduced deliberately (20 objects, `/memories` delayed 1.2 s, page 1 in
  flight when `openMem` fired) and confirmed: 12 cards, newest first, no
  duplicates, and paging still reaches `o01` afterwards.
- **`/ignored` iterated the ignore dict without the lock.** Every mutator
  (`Q` on the key thread, `/unignore`) holds `self.lock`; the reader did not,
  so `sorted(self._ignored.values())` could raise `dictionary changed size
  during iteration` and die as a traceback on the HTTP thread. It now takes
  the lock and snapshots. The guarantee is structural — all real mutators hold
  the lock — not proven by the 300-read churn test, which bangs on the dict
  *without* the lock and is therefore harsher than reality rather than a proof.
- **`forget_view` trusted the id/label pairing the page sent.** A view already
  dropped from another tab, re-dropped, counted as "the last view" and forgot
  the whole object *and its sightings*; and a crafted id could delete any
  point — a sighting, another object's view — while the log said "dropped one
  view". Now the pid must be in `taught_ids(label)`, and otherwise the answer
  is an honest "that view is already gone". Verified against all three cases.
- Two smaller ones: `/rename` had no server-side length cap (the page's
  `maxLength=40` was the only limit, so a crafted POST could write an
  arbitrarily long label onto every point — now capped at 60), and the page
  refused case-only renames, which made a pre-lowercase `"Water bottle"`
  impossible to normalize from the tab. Arming DROP and then tapping the
  picture also used to leave the button armed against a *different* view; the
  picture disarms it now.

## The slowdown, the lock, and the location button

Reported as "the codebase grew and the system slowed down a lot" (2026-08-12,
twelfth round), with a full performance and hygiene review requested. The
review measured first and found the slowdown was not the code — and then
found a real code-side wait the report had not named.

### The slowdown was the hotspot doing NAT for a laptop

Mid-review the operator noticed it himself: his laptop was joined to
`l6-robot` while the robot had Ethernet, so the hotspot's documented
internet-sharing (nm-shared + MASQUERADE, see "Headless appliance") was
routing the laptop's entire internet through the robot's radio — background
sync and all, on the same 5 GHz HT20 link as the ~17 Mbps feed. He
disconnected the laptop and "now its smooth". The robot measured healthy
throughout: 9.7 fps on loopback, CPU ~186%, zero process swap, RSS 4.2 GB.
This is the 2.4 GHz lag story from "Streaming over the hotspot" again, with
the demand coming from a guest this time; TCP queues, the feed falls behind,
the robot looks slow. Fix is operator-facing and now in the README twice
(Headless Appliance, The Feed Is The Bottleneck): don't leave a laptop on the
hotspot. Phones dodge it by preferring cellular. Disabling the passthrough
was considered and not done — it is also the maintenance path, and a fixed
warning beats machinery.

### What the review did find: tab reads queued behind YOLO for no reason

Measured on the live service: `/state` (no lock) answers in 10 ms median;
`/memories` took **809 ms median, 1.6 s max**, and `/ignored` — a trivial
dict read — 0.45–1.6 s. But the scan `/memories` runs costs **0.5 ms** on
this shard (benchmarked on a copy). The entire difference was waiting for
`self.lock`, which the detect thread holds for a whole YOLO pass — and lock
wakeups are not FIFO, so a waiter can lose more than one round. Every tab
open, delete and rename paid it, twice (both endpoints), because every
mutation redraws the tab.

Two changes, both making reads not need the big lock rather than making the
lock free faster:

- **`Memory` now serializes its own shard access** with an internal `RLock`
  (every method that touches `self.shard`, including `reopen`/`close`, which
  were the documented reason the tab needed the app lock at all). The
  app-level lock still serializes compound robot mutations exactly as
  before; `memories()` just no longer takes it. RLock because compound reads
  nest (`objects` → `count_label`). Lock order is always app lock → memory
  lock, never the reverse, so no deadlock is possible.
- **`Detector._ignored` is copy-on-write**: mutators replace the dict, never
  edit it, so `/ignored` iterates a snapshot without any lock. Both mutators
  (Q on the key thread, `/unignore` on HTTP) still run under the app lock,
  which is what keeps writers from racing each other.

Verified against a scratch shard: with the app lock deliberately held for
2 s, `/memories` + `/ignored` together answer in **5.3 ms** (was 0.8–1.6 s
each). A 4-reader/1-writer hammer with `reopen()` mixed in raises nothing.

Closed as measured-harmless, so nobody re-suspects them: `memory.count()`
per detect pass costs 0.15 ms even at 6000 points; `objects()` is 24 ms and
`forget`'s full scroll 25 ms at that size (benchmarked on a grown copy — the
harness is `bench/bench.py` in the session scratchpad). Shard growth is not
what anyone is feeling. Dependency and import hygiene also came back clean:
nothing unused in `pyproject.toml` (matplotlib arrives via ultralytics), no
dead imports, no dead functions found.

### SET LOCATION — the tab edits where the robot is

`--location` was the only way to set the place stamped on memories, which on
the appliance means editing the service file. The memory tab's first row now
shows `here: <place>` with SET LOCATION / CHANGE; it POSTs `/where`, which
normalizes and caps at 60 chars server-side exactly like `/rename` and for
the same reason. Empty clears it. `Robot.set_where` logs it; the banner
confirms it.

- **Persisted in `edge-data/where.txt`**, read at `Memory` init. Beside the
  shard, not in `.env`: it is state that travels with the memories, not
  calibration. The appliance's off switch is a power cut, so a
  session-only value would silently revert to the service file's idea of
  home and stamp a booth's memories with the living room.
- **Precedence: `--location` still wins for the run** and does NOT rewrite
  the file — consistent with the config philosophy (flag = one-run A/B).
  The button writes both the live value and the file.
- Old points keep the place they were written at. That is the feature —
  "where are my keys" answers where it saw them — and it is why this is a
  stamp, not a payload migration.
- The new text field inherits the global-keydown trap recorded under "People
  are objects now"; the existing INPUT-target guard covers it, and the
  browser probe proves typing in it fires no `/key` calls. One new trap of
  its own: a page landing from an infinite-scroll fetch redraws the row, so
  `drawLoc` is skipped while the field holds focus or it would throw away
  the operator's typing mid-word.

Verified: server-side persistence/precedence/normalization plus the
2-second-lock-hold latency test (`bench/test_where.py` in the session
scratchpad, against a scratch copy — never the live shard), score parity
byte-identical (same-object min 0.887 / med 0.920, different med 0.472 /
max 0.611, margin +0.275), replay verdicts unchanged, and a headless-firefox
probe of the real page against a stub server driving the full edit flow.

## Recall answered from the wrong space

Reported (2026-08-12, thirteenth round): *"Where did I leave my hat?"*
answered **dylan**, with hat second; the operator also asked why the panel
shows a SEEN and a HEARD list when the heard entries are just the teaching
sentences, and proposed answering with the last three sightings of the best
match. He was right on all three, and the first one is measurable.

### Why dylan beat hat: the ranking was noise

Recall embedded the whole question with CLIP's TEXT tower and searched the
IMAGE vectors of everything seen today. Reproduced on a shard copy with the
operator's exact question:

| space searched | top scores |
|---|---|
| CLIP text → image vectors (what shipped) | 0.262, 0.259, 0.258, 0.256… — hat and dylan interleaved in one flat 0.246–0.262 band |
| Nomic question → taught transcripts | **hat 0.725**, dylan 0.424, mirror 0.388 |

CLIP text-to-image over a full interrogative sentence separates nothing;
the daily winner is lottery, and the lottery favors whoever has the most
tickets — dylan had 62 sightings that day, so dylan often topped it. The
text space, by contrast, is language-vs-language and lands on the right
object by a 0.3 margin even against bare-noun transcripts ("Hat.").

### What recall is now

`Robot.ask` is two steps: **pick the object in the text space**
(`Memory.best_taught`, question vs taught transcripts), then **answer with
its newest sightings by time** (`Memory.last_sightings`) — a scroll and a
sort, no vector search, because once the object is chosen "where did I
leave it" is a question about time, not similarity. Three rows, each with
time, place and the logged crop; the spoken line reads them out ("I saw hat
at 10:44 PM. Before that at 10:34 PM and 10:22 PM."). Not day-filtered,
deliberately: keys left yesterday are the whole use case, and `_when()`
names the day for anything older than today.

- **The rows are episodes, not log lines.** A sighting is written per track
  birth, so an object sitting in view logs bursts — the raw newest three
  read "at 10:42, before that at 10:42 and 10:42" (observed). `last_sightings`
  keeps the newest of each burst, minimum 600 s apart (`apart=`).
- **"What did you see today?" stays**, as an inventory rather than a
  search: distinct labels sighted since morning (`Memory.seen_since`), no
  vectors at all. Same phrase check as before, same card shape.
- **SEEN/HEARD is gone from the panel** — one list. The heard column was
  the taught transcripts (only taught points carry text vectors), i.e. the
  definitions, as the operator said.
- **CLIP's text tower is deleted** (`_clip_text`, `embed_query_clip`,
  `CLIP_TEXT_MODEL`), since this was its only caller. `warm_encoders` lost
  its `"t"/"a"` split — both actions now need the same two models — and
  first-ask load drops by that model's ~2.5 s. `headless-setup.sh`'s cache
  pre-fill was the easy-to-miss second caller (the `verify_scores.py`
  lesson again: grep for other builders before deleting a loader).
- `day_recall` and `_dedupe_by_label` are deleted with it. If anyone wants
  the cross-modal image-space search back as a course beat, it is one git
  revert away — but re-read the table above first: it was measured not
  working, on the very question the demo script asks.
- The empty-shard answers are honest now: "You haven't taught me anything
  yet." / "I know hat, but I haven't seen it around." instead of "I didn't
  see anything like that today."

Also observed while investigating, distinct problem, no code change: one of
dylan's three taught views is a picture of the background with dylan gone —
a stale-box teach (the one-blink window under "UI responsiveness", or a
walk-away during the hold). A background view wearing a person's name is a
memory that matches empty desk, which both inflates that label's sightings
and feeds it junk crops. The cure is the button built for it: open the
card, tap the picture until the background view shows, DROP THIS VIEW.

Verified: `bench/test_ask.py` (session scratchpad) against a scratch copy —
right label at 0.72+, rows newest-first and ≥600 s apart, inventory
distinct, empty-shard refusals; score parity and replay byte-identical to
the pre-change run; browser probe of the new card against a stub server.

### Follow-ups from first use, same evening

Four operator reports after driving the new recall from the phone:

- **Sightings now store the scene-style picture** (box + margin, unmasked —
  `Robot._scene`), not the masked gray recognition crop: recall shows these
  rows to a human, and "where did I leave it" answered with a fragment on a
  gray void. Same payload key, so old sightings keep their old crops with
  no migration; they just look gray next to new ones, exactly like
  pre-scene taught views do in the tab.
- **The recall card's rows grew up**: the object's name above a 150 px
  4/3 `contain` picture (`.sight` in the CSS), same reasoning as the tab's
  cards — `cover` cuts the ends off the object the picture exists to show.
- **SET LOCATION zoomed the page in on iOS and never zoomed back.** iOS
  Safari zooms onto any focused input whose font-size is under 16px, and
  does not undo it when the field goes away. The rename field never
  triggered it (.edit is 17px); the location field's 14px override did.
  It is 16px now, and the comment above it says why: **any input added to
  this page needs ≥16px font**, or it inherits the bug.
- ~~**"Why are ignored items deleted?" — they aren't.**~~ The answer given
  here (session-only by design, nothing stored) was accurate and did not
  survive contact with the operator, who reported it again the next day as
  a bug. Round 14 makes dismissals persistent — see "The prompt, the
  clutter that came back, and the photos the tab couldn't show".
- And one more lock the measurement said to drop: **`ask` no longer takes
  the app lock** in `_process` — it reads only memory (which serializes
  itself) and touches no track state, so the app lock bought nothing but up
  to a second of queueing behind a YOLO pass on an answer the operator is
  watching for. The remaining ~6 s of ask latency is Whisper's documented
  floor ("Teach latency" above): do not chase it.

### Found by adversarial review of rounds 12–13, one fixed

- **`where.txt` broke the "does a shard exist" heuristic from inside the
  same `__init__` that reads it.** `Memory` treated ANY file in the data
  dir as proof a shard exists and called `EdgeShard.load` — so a dir
  holding only `where.txt` (segments hand-deleted, a restore that died
  early) crashed every boot, presenting as a `Restart=always` loop on a
  headless box. `thumbs/` had the same latent property all along; the
  location file just made it two producers. Fixed by excluding the app's
  own two benign files from the check, keeping the fail-loud behaviour for
  anything unrecognized. Verified both directions on scratch dirs. Side
  effect worth knowing: a hand-wiped shard dir now boots fresh and keeps
  its location.
- Noted, deliberately not fixed — all pre-existing, all already documented
  where they live: the `busy` check-and-set is still non-atomic (one
  operator, actions seconds apart); `banner`'s seq bump is not atomic
  across its many writer threads (worst case is one banner shown late, on
  a race measured in bytecodes); `startswith` routing still depends on
  `/forget_view` preceding `/forget`; `count_label` is exact-case while
  everything else case-folds (a cosmetic count, wrong only on unmigrated
  pre-lowercase shards); and `_one_per_label` reading `focused()`'s
  incumbent is the documented display-filter trap, enforced by prose, not
  code. Each is a real cost the smallest-change rule keeps choosing to pay;
  fix them when one of them actually fires, and update this list when you
  do.

## The prompt, the clutter that came back, and the photos the tab couldn't show

Three phone reports (2026-08-13, fourteenth round): the mic permission prompt
on every use, "ignored items get deleted each session", and the memory tab
showing only manually taught views — *"When did you last see Dylan" shows
different views than the Memory tab. The memory tab should show all the
views.* One was documentation, two were code.

### The mic prompt is the browser's, and no code path causes it

Read before changing anything: the page requests the microphone **once per
page load** (`mic()` memoizes the promise; see "The voice path") and keeps
it. The per-use prompt is two browser behaviours multiplying: the default
per-site mic permission is *Ask*, which re-prompts on every fresh page load,
and a phone reloads this page constantly because backgrounding the tab or
locking the screen evicts it. The fix is a one-time phone setting, now in the
README ("Stopping The 'Allow Microphone?' Prompt"): iOS Safari **aA →
Website Settings → Microphone → Allow**; Android Chrome remembers the grant
on its own **only once the certificate is trusted** — Chrome refuses to
persist permissions for an origin whose certificate it distrusts, so an
uninstalled cert is what makes its prompt come back. Do not add machinery
for this; there is no web API that persists a permission.

### Ignores persist as vectors now — the tid was never storable

The old design stored a track id, and ids restart from 1 every run, so there
was *nothing to persist*: a saved tid would block whatever unrelated track
inherits the number next boot — the exact tid-reuse trap "UI responsiveness"
documents. The durable name an ignored object has is what it looks like. So
`Q` now writes a `kind="ignored"` point (image vector + scene thumb,
`Memory.ignore`), and `process_frame` re-suppresses by appearance: an
unknown whose requery matched **no taught view** is checked against ignored
points (`Memory.match_ignored`, same 0.90 bar) and silently re-dismissed on
a hit — tid-blocked via `Detector.ignore(tid, pid=...)`, not displayed, not
teachable. The detector's `_ignored` dict is demoted to a per-frame gate;
each block carries the pid it came from so `unignore(pid)` (which deletes
the point after verifying the pid *is* an ignored point — the `forget_view`
lesson) can lift its live blocks too via `Detector.unblock`.

Order matters and is load-bearing: **taught always beats ignored**, because
the ignore check runs only when recognition returned nothing. A degenerate
ignored crop — the blank/reflective kind that matches everything (0.951
between two gray crops, see "Measured baselines") — can therefore at worst
hide unknown *clutter*, never a taught object. That residual failure mode is
real and accepted: if an ignore ever swallows an unknown someone wanted, it
is visible in the tab's IGNORED list with its picture, and TRACK AGAIN
undoes it. The tab lists from the shard now (`/ignored` reads `Memory`, no
app lock, same reasoning as round 12), and `/unignore` takes `pid=` — **as a
string in JSON**, the same 2^53 trap every point id on this page has.
Side effects worth knowing: dismissals now survive REBOOT too (the shard
reloads, the next requery re-suppresses), and a restart no longer empties
the IGNORED list — the round-13 note explaining that emptying is struck
through above.

### The tab's card now shows the robot's own photos

Sightings were only reachable through recall's answer card. `Memory.objects`
now attaches each label's recent sightings (`last_sightings`, limit 8 — the
same burst-collapse recall uses, so eight rows are eight occasions), and the
card's tap-cycle walks taught views first, then sightings, captioned
`view 1 of 2` vs `seen 1 of 8`. DROP THIS VIEW hides itself while a sighting
shows (`display:none` toggled in `show()`): sightings are history, not
taught memories, and the server would refuse the id anyway since
`forget_view` checks `taught_ids`. Eight is a cap, not "all" — the operator
asked for all the views, and all *distinct occasions* is what makes sense: a
mug sitting in view logs hundreds of near-identical bursts. FORGET still
deletes sightings with the object, unchanged.

### Verified

`bench/test_ignore.py` (session scratchpad) against a scratch shard — 20
checks through the real `Memory` and the real `process_frame` (stub
detector wearing the real `ignore`/`unblock` methods, stubbed embeds):
persistence across reopen, re-suppression of a matching unknown, a stranger
still offered, taught-beats-ignored with a trap point at the taught vector,
unignore lifting live blocks, pid-door refusing a taught pid, and the
sightings grouping/burst-collapse. `bench/stub_probe.py` drove the real
page in headless firefox against a stub server (never the live service):
the 5-step tap-cycle with correct captions, DROP visible on taught views
only, no transcript on sightings, cycle wrap-around, and TRACK AGAIN
posting the >2^53 pid as the exact string. Headless replay over `testdata/`
is byte-identical before/after, and `verify_scores.py` parity is unchanged
(same-object min 0.887 / med 0.920, different med 0.472 / max 0.611,
margin +0.275).

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
- Live app: 2.3 GB / 18 threads at startup; 3.4 GB / 34 threads after one ask;
  drifts to ~4.2 GB over a day of interactions (shard, caches — not a leak
  anyone has chased). Add one thread for the key handler (see "UI
  responsiveness"). Swap is active again as the OOM backstop; steady-state
  use of it should stay 0.
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
