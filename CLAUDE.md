# L6 robot — working notes

Teaching material for L6 of the course: a robot that sees objects, learns them
by voice, and recalls where it saw them. It runs on a Jetson Orin Nano as a
headless appliance on its own Wi-Fi, driven from a phone. **Solved, shipped and
in daily use.** Nothing here is unproven; cold boot included.

This file is the durable handoff — the decisions, constants and traps that cost
real time to find. Update it in the same commit as the change it describes.
Sixteen rounds of work are compressed here; the full narrative of each is in
`git log` if you ever need it.

## How to work on this repo

- **Prefer the smallest change that works.** This is beginners' teaching
  material; extra machinery costs more here than a few hundred MB. Four ideas
  have been built and then deleted for failing that test (a static crop-quality
  gate, a scene-relative teach gate, `MAX_EMBEDS_PER_PASS`, a focus dwell
  timer). Don't rebuild them without a measurement.
- **Measure on the device before changing anything.** Almost every "obvious"
  cause in this project's history was wrong: the slow feed was startup memory,
  not CUDA; the promiscuous matching was crop scale, not Qdrant; the slow
  network was a laptop NAT-ing through the hotspot; the 6 s teach floor was a
  library running its model twice. Throwaway harnesses go in the session
  scratchpad, not the repo.
- **Never test against the live shard.** Point `--data` at a scratch copy. A
  scripted teach against a blank crop poisons memory in a way that mimics a
  threshold regression, and one wrong browser probe would have rebooted the
  operator's shard mid-demo.
- **Two checks to re-run after anything near detection, crops or encoders:**
  `uv run python testdata/verify_scores.py` (score parity, below) and headless
  replay `--source testdata/`. Both should be byte-identical unless you meant
  to move them.
- **Adversarial review of the diff has found more real bugs than testing has**,
  every round, and none of them showed up on the bench. The trap list below is
  the accumulated output; read it before touching the area it names.
- Prefer official NVIDIA docs and staff forum guidance. Do not guess wheel URLs.
- Preserve user changes.

## Goal and constraints

- Target: Jetson Orin Nano Super 8 GB, Ubuntu 24.04/aarch64, Jetson Linux
  R39.2 / JetPack 7.2, NVMe, original NVIDIA 19 V supply.
- **Power mode stays 15 W.** Do not change fan controls, over-current limits or
  the device tree, and do not use `jetson_clocks`/MAXN. None are needed.
- **Preserve the model stack and the pinned FastEmbed/onnx-asr versions.** The
  recognition threshold is calibrated against those exact embeddings *and* the
  live crop pipeline, so any change to encoders, providers or crops needs a
  score-parity re-run.
- The repo mirrors the course's helper.py: Nomic 768 (text), CLIP 512 vision
  (image), Whisper (speech). CLIP's text tower was deleted — see "Recall".

## Repository state

- `/home/qdrant/Documents/github/l6-robot`. PRs #1–#10 merged to `main`.
- Branch `codex/persistent-ignore-and-sightings` (PR #11): rounds 14–16
  (persistent ignores, the maybe band, the ASR speedup). Deployed and serving.
- Branch `codex/remote-access-doc`: `REMOTE-ACCESS.md`, docs only — how to get
  a shell on the headless unit over three independent routes plus serial, and
  how to move it between Wi-Fi networks. Records that `tmux` was installed.
- `.env` is gitignored; **`.env.example` is the documentation for every
  per-camera knob**. `robot.log` is untracked (`*.log`).
- Operator-facing docs live in `README.md`; keep it in sync rather than
  duplicating it here. Machine setup is `deploy/headless-setup.sh` (idempotent,
  safe to re-run live) and `deploy/l6-robot.service`.

## Recognition: the threshold, and why it does not move

`RECOGNIZE_THRESHOLD = 0.90` in `.env`. **CLIP cosine has a high floor** — two
unrelated crops from the same camera routinely score 0.75–0.85 — so this is a
knee in a distribution, not a confidence. It is high because the app's crops
are small (as little as 36x95 px) and mask-filled, and resolution is the
dominant factor in the score: the same pairs score 0.928/0.479 as plain studio
crops and 0.948/0.611 masked and shrunk to live scale.

**0.88 was measured and REFUSED.** Read this before touching the bar again. On
the live shard (809 sightings, 8 taught views): 0.90 gives 768 recognized / 44
mislabelled; 0.88 gives 775 / 49. It buys 2 correct recognitions and 5
mislabels — and **per taught view, the number of its own sightings missed is
identical at both bars**, because the misses sit at 0.75–0.81, nowhere near
it. The evidence that looked like a knife-edge was a *mirror* scoring
0.86–0.90 against taught objects, i.e. exactly the promiscuity 0.90 exists to
prevent. **Lesson: a score series near the bar means nothing until you know
whether it belongs to an object that should match.**

**The real recognition fix is coverage, and it needs no code.** A label's
misses are a coverage problem: the laptop's best single taught view covered
103 of its 173 sightings; the union of its 6 views covered all 173. Teach each
object 2–3 times, turning it between teaches. This is in the README.

Closed, do not reopen:

- **The Qdrant layer is clean.** Verified empirically against the live shard:
  stored vectors have L2 norm 1.000000 (Qdrant pre-normalizes for Cosine),
  shard scores match hand-computed numpy cosine to 1e-7, named vectors
  correctly wired (`image` 512, `text` 768, not swapped).
- **Do not filter crops; calibrate the threshold.** A static quality gate
  predicts promiscuity poorly (short side −0.74 Spearman, sharpness −0.60,
  mask-fill fraction *noise* at +0.08), and the crop that broke the demo was
  141x171 with Laplacian variance 553 — neither small nor blurry. A
  scene-relative gate was built and cut: at 0.90 that same crop claims 6% of
  the room. Raising the bar neutralized it on its own. A degenerate crop is
  only degenerate *relative to the operating threshold*.
- Two featureless gray crops score **0.951** against each other. Teaching a
  blank or background crop makes a memory that matches everything. Nothing
  rejects one; the calibrated threshold is what keeps it harmless.

### The maybe band

A score within `MAYBE_MARGIN = 0.05` under the bar surfaces as an orange
`hat? 0.87` box and panel row instead of a bare UNKNOWN. **Display only, and
that is load-bearing**: a guessed track is still unlabeled — still the teach
target, writes no sighting, cannot be forgotten, still checked against the
ignore list. `Robot.forget` strips matching guesses off live tracks like it
strips labels. Showing the band is the *opposite* move to lowering the bar: a
hat at 0.87 tells the operator to teach that angle, and a mirror at 0.87 shows
a spectator what the threshold is keeping out. If anyone proposes making a
guess *act* — auto-label, auto-sight, a softer FORGET — re-read the 0.88
measurement above.

The one sanctioned act on a guess is the operator confirming it: tapping the
orange label POSTs `/confirm` with the **displayed** label, and `Robot.confirm`
re-verifies it against the live guess before teaching. That check matters —
focus moves under a finger already on its way down.

## Detection, attention and the constants

`robot/detect.py`, all field-tuned; each does one job and they are easy to
confuse.

| constant | value | what it decides |
|---|---|---|
| `DETECT_CONF` | 0.30 (`.env`) | proposal confidence floor; the free knob against a blinking box |
| `DETECT_MIN_AREA` | 0.001 (`.env`) | smallest proposal kept — raise against small far-away clutter |
| `DETECT_MAX_AREA` | 0.20 (`.env`) | largest kept; drops wall/desk/torso phantoms |
| `STABLE_FRAMES` | 3 | passes before a track is displayed, embedded or teachable |
| `REQUERY_SECONDS` | 2.0 | how often a stable track re-asks memory |
| `DEAD_SECONDS` | 5.0 | how long a track survives with no detection **as itself** |
| `FOCUS_MARGIN` | 1.6 (`core.py`) | attention hysteresis |
| `PAD` / `SCENE_PAD` | 0.12 / 0.3 | recognition crop margin / the human-facing picture's margin |

Both area knobs are **areas**, so they move as the square of apparent size:
0.0008 → 0.001 is ~12% wider, not 25%.

- **`DEAD_SECONDS` is not the box's lifetime.** The box's lifetime is the
  `frames` decay (`max(0, frames - 1)` on a miss, capped at `STABLE_FRAMES + 1`
  on the way up), which hides a box after two missed passes regardless. This
  only decides whether the object comes back *as itself*. It was 1.5 s, which
  was far stricter than BoT-SORT's own 30-frame buffer underneath it, and a
  real object's confidence swings either side of `DETECT_CONF` (a microphone
  measured 0.34–0.91, absent on some frames) — so routine dropouts destroyed
  the `Track` and its label, note, thumb and vec, and the object returned as a
  stranger. That was the "comes and goes" on a static scene: 57 `Track` births
  across 17 tracker ids in 25 minutes. At 5.0 s, births went 4.12 → 2.33/min.
  **The visible blink is unchanged** (68 box disappearances per 130 s before
  and after) — that is the `frames` decay, and the way to fix it is steadier
  detection, not this constant.
- **The `frames` cap is one above the gate deliberately.** Every extra frame of
  credit is a frame in which a departed object still has a box and a **stale
  crop**, and is still teachable — whip an object away, press TEACH, and you
  teach a picture of where it used to be, which is the blank-crop poisoning
  above.
- **`FOCUS_MARGIN` is the one number to turn** if the box feels twitchy. It
  went 1.25 → 1.6 on an operator report plus a measured 3.5 focus switches/min
  (median dwell 2.34 s); after, 0 in 120 s. Raising it makes focus harder to
  move *on purpose*, which is its own failure, and **nobody has measured a
  deliberate takeover at 1.6.** A dwell timer stays rejected: any hold gated on
  candidacy is powerless against a blink (`focused` can only retain an
  incumbent still in the candidate list, which is stable tracks only), and a
  hold that returned a non-candidate would aim the panel and TEACH at an object
  with no box and a stale crop.
- **Attention and the teach target are decided once per detect pass**, inside
  `Robot.process_frame`, and published as `Robot.attention` (panel subject,
  FORGET target) and `Robot.teachable` (the salient unknown, TEACH target).
  Single writer, on the thread that owns track state. The front end reads them.
- **`_one_per_label`**: at most one box per remembered object, best score then
  salience. Teaching yourself twice means the person box, face box and torso
  box all come back "dylan"; one memory, one box. Losers keep their labels and
  tracking, are simply not drawn and **not written as sightings**. Accepted
  cost: two identical mugs in frame show one box between them.
- **Nothing reads a detector class name anywhere in the app.** Detection finds
  *a thing*; memory decides *which* thing. The one place that consulted the
  detector's vocabulary was the person blacklist, which is deleted — people are
  ordinary teachable objects now. A seated person at desk distance is
  0.072–0.078 of the frame, comfortably inside `DETECT_MAX_AREA`; a face
  filling the view is not. CLIP is not a face recognizer: 512 dims off a 224 px
  crop keys on the whole silhouette, clothing included. Untested with more than
  one person.
- **The lens has visible barrel distortion**, so a crop's geometry depends on
  where in frame it sits, which biases both scores and `salience`. Undistorting
  is `calibrateCamera` + a ~3 ms `remap`, but it **invalidates every existing
  memory** and forces a re-teach. A deliberate project, not a flicker fix. Free
  mitigation: teach and demo near the centre of frame.
- `MAX_DET = 64` is not being hit — YOLOE emits 9–10 proposals per frame. About
  8 of ~10 stable tracks are embedded and discarded every 2 s, which is real
  waste and explains nothing anyone sees (passes measure 0.1–0.6 s).

## The voice path

Whisper-base via onnx-asr, on CPU. **The teach beat is ~1.9 s warm**, and
getting there took undoing two wrong beliefs, so read this before optimizing.

- **`LANGUAGE = "en"` in `robot/models.py` halves every action.** This Whisper
  is the beam-search export: one ONNX graph holding the encoder *and* decoder.
  With no language given, `recognize_batch` runs that whole graph a **second
  time** just to read the language token. Measured on a real 2.1 s utterance:
  5.37 s auto vs 2.71 s named at 2 threads, byte-identical transcript. It also
  stopped auto-detection storing an English teach as `делан`. Set it to another
  Whisper language code to teach in that language, or `None` to pay the second
  pass.
- **`ASR_THREADS = 4`, separate from `ENCODER_THREADS = 2`.** Two is right for
  Nomic and CLIP, which the detect thread calls forever beside YOLO. It is
  wrong for Whisper, which runs only while a human waits and never beside
  another encoder. Live app, release→answer warm: 2.82 s at 2, **1.90 s at 4**,
  and the feed is identical to idle either way (102 ms median gap, 132 ms max,
  no stall over 400 ms in 20 s). Do not raise past 4 without re-running that
  harness: six cores, and the pump needs one.
- **`trim_to_speech` cuts silence off both ends before transcribing.** Whisper
  charges for every second it is given *and* hallucinates repetitions on
  silence: a 7.7 s hold holding 1.4 s of speech transcribed to `L L L L L…` in
  16.3 s; trimmed to 2.0 s it gave words in 5.2 s. It fails open, and there is
  a deliberate gap — `is_silent` passes anything at rms 120, the trim needs a
  100 ms block at `SPEECH_RMS` 200, so a very quiet hold is neither rejected
  nor trimmed. **Do not close it by lowering `SPEECH_RMS`**: `record_wav` stops
  recording on that same number, and a lower bar means the laptop path stops on
  room noise.
- **Whisper-base mishears bare nouns** ("laptop" → `La-caw`). Not gain —
  normalizing to 50% and 90% returned `La-caw` both times. The fix is
  operator-facing: say "this is my laptop", which is also what `parse_label`
  reads. int8 quantization was measured (0.98 s at 4 threads) and **refused**:
  it is the only change that moved the words, and bare nouns are already this
  model's weakest point.
- **Labels are lowercased in `parse_label`** — the single door every label
  enters through. They are exact-match keys (forget deletes by string, recall
  dedupes by string), and Whisper capitalizes on a whim, so `Laptop` and
  `laptop` were two objects. The three read-side comparisons (`Memory.forget`,
  `_dedupe_by_label`, the strip loop in `Robot.forget`) are case-folded too, so
  old shards are fixed with no migration.
- **`warm_encoders` is called from two threads per interaction** — a background
  warm at phone pointerdown, so the ~7 s of first-session model loads run under
  the button hold, and again in `_process` before transcribing and before the
  lock. `_warm_lock` is required: **`lru_cache` does not serialize a cache
  miss**, so both would build the same model at once. Being under the hold is
  *relocation, not removal* — a cold load holds the GIL in 1.4–1.6 s blocks and
  every thread pauses with it (a 20 ms ticker lost 75–89% of its wakeups). It
  is safe only because a ~3 s hold sits inside `DEAD_SECONDS = 5.0`; that
  margin is load-bearing and is not large.
- **The laptop path deliberately does NOT warm at press.** `record_wav`
  re-enters `stream.read` every 100 ms, and a 1.6 s GIL hold overflows
  PortAudio's buffer — the first laptop teach of a session would record chopped
  audio and fail looking exactly like a Whisper bug. `record_wav` now prints a
  line when the overflow flag is set.
- **The browser mic is opened once per page and kept.** `mic()` memoizes the
  **promise**, not `ctx` — `ctx` is assigned after three awaits, so an
  `if (ctx) return` guard lets the warm-up listener and the first press both
  through, and two worklet nodes on one chunk list records the audio twice.
  The context asks for 16 kHz (Whisper's rate) so the browser resamples instead
  of pushing 3x the bytes up a half-duplex link; browsers that ignore it report
  the real rate in `ctx.sampleRate`, which the WAV header is written from. A
  kept mic needs recovery a per-press one got free: `start()` resumes a
  suspended context (backgrounding a tab suspends it; an iOS call does it), and
  `onended` clears the memo so the next press opens fresh. The permanent
  recording indicator is the price, and it is in the README.
- **The "allow microphone?" prompt on every use is the browser, not the app.**
  Chrome on iOS re-prompts on every press while the origin's certificate is
  untrusted, and iOS only installs profiles **Safari** downloads. There is no
  web API that persists a permission. README covers it; add no machinery.
- Speech-to-text runs **before** the live-state lock is claimed, so the
  detector keeps tracking. `ask` takes no app lock at all (it reads only memory,
  which serializes itself). `Robot.teach` takes the transcript, not the WAV.

## Memory and the shard

One Qdrant Edge shard, two named vectors (`text` 768 Nomic, `image` 512 CLIP),
cosine. Payload `kind` is `taught` | `seen` | `ignored`.

- **`UpdateOperation.upsert_points` with an existing id does NOT rewrite the
  point** in this qdrant_edge build — it reports nothing and changes nothing.
  Use `UpdateOperation.set_payload(ids, payload)`. This has bitten twice.
- **Every write flushes.** That is what makes pulling the plug the documented
  off switch, and shards have survived several aborted shutdowns intact.
- **`Memory` serializes its own shard access** with an internal `RLock`
  (RLock because compound reads nest: `objects` → `count_label`). This is what
  lets the memory tab read without the app's live-state lock, which the detect
  thread holds for a whole YOLO pass — the tab used to wait 0.8–1.6 s for a
  half-millisecond scan. **Lock order is always app lock → memory lock**, never
  the reverse.
- **`SIGHTING_APART = 600` is named once and shared** by the burst-collapse in
  `last_sightings` and by `forget_sighting_burst`. They must agree: a display
  window wider than the delete window resurrects "deleted" rows. A sighting is
  written per track birth, so an object in view logs bursts; the tab and recall
  show one row per *occasion*, and a drop deletes that row's whole burst.
  Deleting only the shown point promotes a near-identical frame into the same
  slot — measured, and it reads as a delete that did nothing.
- **`forget_view` and `unignore` verify the id against the shard** rather than
  trusting the page. A view already dropped from another tab would otherwise
  count as "the last one" and forget the whole object, sightings and all; a
  crafted id could delete anything while the log said otherwise.
- **Ignores are vectors, not track ids.** Track ids restart from 1 every run,
  so there was nothing to persist. `Q` writes a `kind="ignored"` point, and
  `process_frame` re-suppresses by appearance at requery. **Order is
  load-bearing: taught always beats ignored**, because the ignore check runs
  only when recognition returned nothing — so a degenerate ignored crop can
  hide unknown *clutter* at worst, never a taught object. The detector's
  `_ignored` dict is only a per-frame tid gate; each block carries the pid it
  came from so deleting the point lifts its blocks (`Detector.unblock`). It is
  **copy-on-write** — mutators replace the dict, so `/ignored` can iterate a
  snapshot with no lock.
- **`Robot.ignore` clears `attention`/`teachable`** and the key handler drops
  the track from `LiveApp.tracks`, so a dismissed object's box goes on the next
  composed frame instead of at the next detect pass. Same shape as
  `Robot.forget` stripping labels itself instead of waiting for a requery.
- **Two pictures per memory, and only one is scored.** The embedded crop is
  masked and gray-filled (`padded_crop`); the `scene` picture is the detector's
  box plus `SCENE_PAD`, unmasked, 480 px, JPEG 80 — it exists to be *looked
  at*, in the memory tab and in recall's answer. Points written before it
  exists fall back to the crop, which is why old memories look gray beside new
  ones and why nothing needed migrating. **Scene pictures contain whatever was
  in frame, including people**; worth a thought before a shard travels.
- The scan-everything reads (`objects`, `forget`, `taught_ids`) are full
  scrolls. Benchmarked on a grown copy: `objects()` 24 ms and `forget` 25 ms at
  6000 points, `count()` 0.15 ms. **Shard growth is not what anyone is
  feeling.** If a shard ever holds hundreds of objects, page the scroll itself.

### Recall answers from the text space

`Robot.ask` is two steps: **pick the object in Nomic's text space**
(`best_taught`, question vs taught transcripts), then **answer with its newest
sightings by time** (`last_sightings`) — a scroll and a sort, no vector search.
Once the object is chosen, "where did I leave it" is a question about time, not
similarity.

The old version searched the image space with CLIP's text tower and the whole
question. Measured on the operator's own question: that put every sighting in
one flat 0.246–0.262 band, so the winner was whoever had the most sightings
that day ("where did I leave my hat?" → **dylan**). The text space lands on
`hat` at 0.725 against 0.424 for the runner-up, even against bare-noun
transcripts. **CLIP's text tower is deleted**; a revert is one commit away, but
re-read that table first. Not day-filtered, deliberately: keys left yesterday
are the use case. "What did you see today?" stays as an *inventory* — distinct
labels since morning, no vectors.

## Threads, locks and the page

The frame pump is the one thread that must never block: it does `cap.read()`
(103 ms), draws boxes and encodes the JPEG the browser is streaming. Work that
used to sit on it — key handling, thumbnail reads, panel composition — is why
the UI stuttered. It now captures and renders and **takes no lock at all**.

- `_key_loop` is one dedicated thread, deliberately, so two taps on one button
  run in order. It serializes the **key queue only**: phone hold-to-talk
  arrives on HTTP handler threads and is gated only by the non-atomic
  `self.busy` check-and-set. Pre-existing, one operator, shard writes are
  serialized by the lock regardless.
- **No key drain after an action.** `_drain_keys` in `_process` was not just
  dead but dangerous — two consumers on one queue means `get_nowait()` after
  `empty()` can raise, and escaping that `finally` before `busy = False` wedges
  teach and ask until restart. The call in `run()` is fine: it runs before the
  key thread starts.
- **`robot.close()` must hold the lock.** Once FORGET/REBOOT moved off the main
  thread, Ctrl-C during an in-flight forget could close the shard from under a
  scroll/delete/reopen.
- **Never key attention state by `tid`.** `Detector.reset()` restarts tracker
  ids from 1, so a tid-keyed incumbent hands its focus to whatever unrelated
  new track inherits the number — verified. `focused()` holds the `Track`
  object, which makes the whole class impossible.
- **A track holding focus must not be removed from the candidate list by a
  display filter.** `focused` can only retain an incumbent that is still a
  candidate, so a filter that drops it bypasses `FOCUS_MARGIN` entirely and
  re-derives attention by raw salience over what is left — which can be a third
  object. Reproduced with stubs. `_one_per_label` keeps the incumbent's box for
  exactly this reason; anything else that filters `display` inherits it.
- **Teach and forget requery only what they must.** Marking every track
  `last_query = 0` fired a CLIP embed per track in one locked burst. Teach
  stamps the one track it taught (`pending_track`); forget stamps only tracks
  wearing the deleted label. `Track.requery_now()` sets
  `last_query = now - REQUERY_SECONDS` rather than 0 — zeroing makes it falsy
  and blanks the panel to "looking..." for a beat, which reads as a glitch.
- The `/state` panel is **HTML polled at 4 Hz**, not pixels drawn into the
  video. The old composited panel was fixed at 480 px and rendered ~100 pt wide
  on a phone in portrait, which no font tuning fixes. `/stream` carries the
  annotated feed alone.

Page rules, each learned the hard way:

- **`PAGE` is a bytes literal, so it is ASCII-only, comments included.** An em
  dash in a CSS comment is a `SyntaxError`.
- **Never build the panel with `innerHTML`** — labels and notes are whatever
  Whisper heard. Use the `el()` helper and `textContent`.
- **Point ids do not survive JSON in a browser.** They come from
  `itertools.count(time.time_ns())`, ~1.8e18, past 2^53 — the page rounded one,
  asked to delete a point that does not exist, and the view *quietly survived
  its own deletion*. Emit ids as **strings**; anything new that carries an id
  needs the same.
- **The global `keydown` listener fires `/key?k=` for `tarfq`.** Any text field
  on this page inherits the trap: typing "water bottle" sends `a`, `t`, `t`,
  `r` — two voice actions and a REBOOT — and `f` deletes whatever is in frame.
  Guarded by bailing on `INPUT`/`TEXTAREA`/`contenteditable` targets.
- **Any input needs `font-size` ≥ 16px.** iOS Safari zooms onto a smaller
  focused input and never zooms back.
- **`banner` is a property so assigning it bumps `status_seq`.** Keyed on text,
  the page could not tell a stale message from the same message sent again.
- `/state` sends `Cache-Control: no-store`. `protocol_version` stays HTTP/1.0,
  so each poll is its own connection — moving to 1.1 needs an accurate
  `Content-Length` on *every* response or clients hang.
- **Route `/forget_view` before `/forget`** — `startswith` matches both.
- `grid-auto-rows: max-content` on the memory list: with `auto` rows and a
  definite height the browser squeezes rows into it and clips every card's
  buttons, and `align-content:start` does not help because the free space is
  negative.
- A redraw must skip the location row while its field has focus, or an
  infinite-scroll page landing mid-word throws the operator's typing away.
- **Deleting is done by looking at a card, not by aiming the camera.** FORGET
  acted on `Robot.attention` at the instant of the press, and a press is a
  decision plus a hand travelling. The `F` key stays (fastest beat to film, and
  a key is not something a thumb lands on by accident); the button became
  MEMORY. Four buttons fit a phone, five did not.

## The appliance

Apply power and it boots into the robot on its own 5 GHz access point. The five
machine changes and their undo commands are tabled in the README under "What
The Setup Changes"; that table is authoritative. Getting a shell afterwards is
`REMOTE-ACCESS.md`.

- **`SSID` and `PSK` are the only knobs.** The address is deliberately not one:
  `10.42.0.1` is fixed in the script and in the unit's `--advertise`, because a
  hotspot on one address and a certificate naming another is precisely the
  name-mismatch warning this path exists to avoid. An `AP_IP` knob existed
  briefly and was cut for exactly that reason.
- **The Wi-Fi block must stay last** in `headless-setup.sh`, after every step
  that needs internet — it cut the radio before a 1.1 GB download once, and
  would cut an ssh session arriving over Wi-Fi.
- **Re-running the script must not lie about a live AP.** A changed band,
  channel or SSID only reaches the air on a bounce, so the script compares
  `iw dev` against `BAND`/`CHANNEL`/name and prints the `nmcli con up` needed.
  Same class of bug as `ssid` being set only on profile create, which changed
  the password while the radio kept broadcasting the old name.
- **5 GHz channel 44.** 2.4 GHz ch6 carries 15–25 Mbps against a ~13 Mbps
  demand, and TCP answers oversubscription by queueing, not dropping — so the
  feed stays smooth and falls further behind the longer you watch, reading as a
  slow robot. Of the channels this regulatory domain allows an AP to initiate
  on (40, 44, 149, 157), 44 is non-DFS and clear of the 149–165 block venue kit
  crowds into. NM 1.46 has no AP width property, so HT20; that is already
  several times the demand.
- **Don't leave a laptop joined to the hotspot.** nm-shared + MASQUERADE means
  the robot routes that laptop's entire internet through the radio carrying the
  feed. That, not the code, was "the system slowed down a lot". Phones dodge it
  by preferring cellular. The passthrough is also the maintenance path, so it
  stays.
- **`--advertise ADDR` takes an IP only** (enforced in argparse): it is written
  as an `IP:` SAN, a hostname makes openssl reject the certificate, and the
  `except` around TLS setup then serves plain http — the phone mic disappears
  over a typo. `loopback` is derived from `host`, never from the advertised
  address.
- **`--watchdog SECONDS`** (service uses 30) exists for a camera that wedges
  *inside* a driver call, which the frame pump cannot notice because it is the
  stuck thread. Two details it got wrong first: it **arms its own clock**
  rather than waiting for a first frame (a `frame_at is None` guard exempts
  exactly the power-cut USB device the flag exists for, on a box with no
  screen), and the clock is **`time.monotonic()`** — the operator sets the
  appliance's date by hand while it runs, and a stepped wall clock either kills
  a healthy robot or blinds the watchdog for the length of the offset. It
  `os._exit(1)`s deliberately: the lock may never come free, and writes are
  already flushed.
- **The shutdown sequence lives in a `finally:` that starts with
  `signal.signal(SIGINT, SIG_IGN)`.** systemd's `KillMode=control-group`
  signals every process in the cgroup, so `uv` forwarded one SIGINT and the
  kernel delivered another; the second landed inside `server.shutdown()` and
  the process died in a native destructor, exit 134. `KillMode=mixed` would
  have fixed only the systemd half — a human holding Ctrl-C is the same bug.
- **`cache_dir=CACHE_DIR` on every encoder.** FastEmbed defaults to
  `/tmp/fastembed_cache`, and systemd-tmpfiles prunes `/tmp` at 30 days — so
  the first boot after that sweep would hang in a download a robot on its own
  hotspot can never finish. `testdata/verify_scores.py` builds its own
  `ImageEmbedding` and was missed at first; **grep for other model builders
  before deleting or changing a loader** (deleting CLIP's text tower found the
  same trap in `headless-setup.sh`). Verified genuinely offline under
  `sudo unshare -n`, so `HF_HUB_OFFLINE=1` is not needed and would break a
  fresh install.
- `PYTHONUNBUFFERED=1` is in the unit because Python block-buffers stdout into
  a pipe, so journal lines sat in a buffer — and were lost on the abort above.
- **The clock is an open wart.** An offline robot has no NTP and the board
  keeps no time across a power cut without a coin cell on `J3` (`J13` beside it
  is the fan), so a unit switched on cold at a booth stamps memories with the
  time it was packed away. Two fixes in the README; neither is done, because
  the box is on Ethernet and NTP-synced, which hides it.
- Rejected: a physical power button (the kit powers on when DC is applied; the
  header's pin numbers are behind NVIDIA's download login and the same header
  carries force-recovery — **do not fill them in from a blog post**), a
  user-level service with lingering, and automatic fallback between venue Wi-Fi
  and the hotspot (the address would then depend on which branch won, putting
  the certificate warning back on every phone).

### The certificate

**It cannot be removed, only made trustable** — no public CA will sign a
private LAN address. What was actually wrong: the cert carried **no
`subjectAltName` at all**, and browsers stopped reading CN in 2017, so it named
nothing and produced the harsher name-mismatch interstitial that trusting
cannot fix. It was also valid 10 years, which Safari rejects outright (>825
days). Now: SANs for the served IP plus `127.0.0.1`/`localhost`/`<host>.local`,
397 days, and `basicConstraints CA:TRUE` so a phone can install it as a root
(Android will not install a leaf without it). Served at `/cert.crt`, which
gives away nothing the handshake already does. Built with an openssl **config
file**, not `-addext`, which macOS's LibreSSL lacks. Regenerated only when the
address changes (`cert/names.txt`), so a trusted phone stays trusted across
restarts — and on the appliance there is only ever one address.

## Operational notes

- **The Jetson has no audio hardware.** The phone-browser hold-to-talk path is
  the only mic on the appliance; `sounddevice` stays a dependency for laptop
  use. `--host 0.0.0.0` serves HTTPS, which is what lets a phone grant mic
  access at all.
- **Close Firefox and the Codex/ChatGPT Electron app before demoing.** They hold
  ~3 GB and are the only reason swap ever appeared in late measurements.
- `uv sync --locked --dry-run` wants to swap `nvidia-cusparselt-cu13 0.8.1`. A
  pre-existing lock quirk, harmless; `uv run --no-sync` avoids the churn.
- **Replay over `testdata/` covers the crop and recognition path, not the
  attention path.** Those images hold one object each and get a
  `detector.reset()` apiece, so they never exercise focus stickiness or
  `_one_per_label`. Identical replay verdicts are not proof of no behaviour
  change.
- **REBOOT has no button, only the `R` key**, and it does the opposite of what
  its name suggests to an operator — it reloads the shard, it does not wipe it.
  The wipe is `--reset`, deliberately CLI-only so a stray tap cannot erase a
  demo.

## Measured baselines on this board

Re-measure before trusting these after any change.

- YOLOE-11L-seg-pf on CUDA at `imgsz=640`: **~165 ms/frame**, 1.8 GB. Seconds
  per frame on CPU, so CUDA matters.
- Camera (Arducam 1080P, USB 2.0): YUYV 1280x720 at **10 fps**, 103 ms per
  `cap.read()`. This, not the GPU, caps the feed.
- Whisper-base: load ~2.7 s, **transcribe ~1.7 s** for a short utterance at
  `LANGUAGE="en"` and 4 threads. Cost scales with clip length, which dominates
  everything: a 7.7 s hold full of silence cost 16.3 s at the old settings.
- **A whole voice action, release to answer, warm: 1.90 s.** Add ~0.4 s for a
  teach's embeds and shard write. First action of a session ~4.6 s — model
  loads, not speech.
- Nomic load ~4 s then ~0.15 s/embed; CLIP vision embed ~0.16 s.
- Shard: upsert+flush 32 ms, `recognize`/`match_ignored` 0.1 ms, `objects()`
  22 ms, thumb write 5 ms. App live-state lock: median 45 ms, p90 282 ms.
- Live app: 2.3 GB / 18 threads at startup, 3.4 GB / 34 after one ask, drifting
  to ~4.2 GB over a day (shard, caches — not a leak anyone has chased). As a
  systemd service with no desktop: **2.3–2.5 GB / 19 threads, no headless
  penalty**. Clean stop takes 3–5 s, all of it the shard close.
- MJPEG at quality 85: ~210 KB/frame on a cluttered desk, **~17 Mbps** at
  10 fps. Scene dominates size, so re-measure rather than quoting.
  `STREAM_QUALITY` gives 154/134/100 KB at 75/70/60 — turn it down only against
  a measurement. Was 37 Mbps before the duplicate-frame fix: `/stream` pushed
  on a fixed 25 fps tick against a 10 fps pump, so 60% of the bytes were frames
  the browser already had. Now `LiveApp._publish` stamps `self.shot =
  (seq, bytes)` — **one tuple, one bytecode**, so the number and the bytes
  cannot disagree — and the handler sends only on a change, always the latest.
- **Score parity (must hold)**, from `verify_scores.py`, which crops with the
  mask as the app does: same-object min **0.887** / median **0.920**,
  different-object median **0.472** / max **0.611**, margin **+0.275**.
- Diagnose a slow feed in this order, because it separates robot from radio:
  `timeout 5 curl -sk https://127.0.0.1:8765/stream -o /dev/null -w '%{size_download}'`
  measures what the app wants to push with the radio out of the picture; then
  `iw dev wlP1p1s0 info` says which band the AP is *actually* on.

## Over-current (OC3) — expected, not a fault

`oc3_event_cnt` climbs continuously under load while OC1/OC2 stay at zero. OC3
counts *instantaneous* `VDD_IN` spikes caught by a hardware comparator, which
is why `tegrastats` shows nothing near the limit: NVIDIA staff confirm they
cannot be captured by the INA3221 sensor, only counted. Measured here:
sustained 1.9 A against a 4.05 A critical limit, peak 2.27 A, detector latency
flat. The popup applet is suppressed via `Hidden=true` in
`~/.config/autostart/nvpmodel_indicator.desktop`; hardware protection is
untouched (`oc3_throt_en` still 1). Counters:
`cat /sys/devices/platform/soctherm-oc-event/hwmon/hwmon*/oc?_event_cnt` ·
https://forums.developer.nvidia.com/t/oc3-raised-at-current-lower-than-critical/321989

## Settled questions — do not re-litigate

Reopen only with new evidence.

- **The startup-memory fix.** `warm_up()` eagerly built all encoders, and ONNX
  gives every session one spinning thread per core: 3.3 GB and 78 threads
  before the camera opened, ~280 MB free, 2 GB swapped out. Now it loads
  **only the CLIP vision encoder**; the rest load lazily under a button hold.
  2.3 GB / 18 threads / 0 swapped. **A correct configuration does not touch
  swap** — but swap IS the OOM backstop and was found deactivated entirely and
  re-enabled (`swapon /swapfile` + fstab), because `Restart=always` turns an
  OOM kill into a 30 s mid-demo restart.
- **Native `sm_87` arch-list gate: abandoned, and it was the wrong gate.** Stock
  PyPI `torch 2.13.0+cu130` omits `sm_87`, prints a scary warning, and *works* —
  Orin executes the `sm_80` kernels, which is what NVIDIA staff say. The warning
  is cosmetic. `torch`/`torchvision` are pinned explicitly rather than arriving
  via Ultralytics.
- **NVIDIA PyTorch containers: unnecessary.** A derived image ran the whole app
  and passed the strict gate, but the gate does not matter and a container is
  heavy machinery for a beginners' course. Note NVIDIA ends standalone iGPU
  images at release 26.03.
- **Jetson AI Lab SBSA wheels: rejected.** Newest py3.12 aarch64 wheel reported
  only `['sm_110','sm_121']` and failed a real CUDA op. Targets Thor, not Orin.
- **Source-built PyTorch: never needed.**
- **DLA: unavailable on R39.2 and unnecessary.** `libnvdla_compiler.so` is absent
  from the BSP; NVIDIA confirms it. Do not symlink libraries to fake it. The app
  does not use DLA, which is also why a JetPack 6.2.1 downgrade is pointless.
- **MJPG camera capture: tried and reverted.** It does reach 30 fps at 33 ms per
  read, but A/B showed it **doubles the OC3 event rate**, costs ~46% more of a
  core and ~100 mA, for smoothness the detector cannot use at its 0.12 s gate.
- **Subprocess worker for the speech/text encoders: replaced** by lazy loading
  in four lines.
- **Launch-time `OMP_NUM_THREADS` tuning: not needed**; the in-code thread
  settings cover it.
- **GPU Whisper: closed.** This venv's onnxruntime is the CPU-only build, so it
  means sourcing a Jetson ORT-GPU wheel against a pinned stack. The 6 s that
  motivated it is now 1.9 s anyway.
- **Debouncing the buttons: no.** The stutter was work on the wrong thread. A
  double-tap is harmless — the second FORGET finds the label already cleared.
- **No device split in the UI** ("desktop only" for anything): the appliance is
  headless, so the phone is the only UI it has.
- **Browser `speechSynthesis`: considered, not done.** It double-speaks on a
  macOS host driving its own tab, and iOS wants a gesture the round trip has
  outlived. `_answer_line` is split out of `_speak` so the panel shows the
  sentence.

## Known warts, deliberately unfixed

Each is a real cost the smallest-change rule keeps choosing to pay. Fix one
when it actually fires, and update this list when you do.

- `busy` check-and-set is not atomic (one operator, actions seconds apart).
- `banner`'s seq bump is not atomic across its writer threads (worst case: one
  banner shown late, on a race measured in bytecodes).
- `startswith` routing depends on `/forget_view` preceding `/forget`.
- `count_label` is exact-case while everything else case-folds — a cosmetic
  count, wrong only on unmigrated pre-lowercase shards.
- `_one_per_label` reading `focused()`'s incumbent is the display-filter trap
  above, enforced by prose rather than by code.
- An undone ignore orphans its thumbnail (delete removes points, not files).
  5–15 KB on an NVMe.
- Within BoT-SORT's buffer a re-attached id can arrive wearing a stale label
  until its next requery (≤2 s). Reasoned, not observed.
- **No automated test covers the UI.** The headless replay path never renders a
  page. Everything client-side was verified by a throwaway proxy that serves
  the real `PAGE` with a probe appended and screenshots it in headless Firefox
  — against a **stub server on a scratch shard**, never the live service. Two
  traps in that harness: Firefox screenshots on the `load` event, so hold it
  open with a slow image or you photograph a half-laid-out page; and the probe
  needs a way to POST its measurements back, because there is no other way to
  get numbers out.

## If performance ever needs more

Nothing is scheduled; the demo is comfortable. Each needs a score-parity re-run
because each changes the crops.

1. **TensorRT FP16 export of the current YOLOE checkpoint** — officially
   supported, likely the best memory/latency win, removes PyTorch from
   steady-state inference. Prompt-free export, masks, tracking and on-device
   engine build all need validating. Do not start with INT8.
2. **A smaller YOLOE segmentation checkpoint** — less memory and compute, but it
   changes which objects and masks appear. Closed-class COCO alternatives only
   propose 80 categories and will miss arbitrary demo objects; FastSAM-s is the
   closest fit to "find a thing, discard the label".
3. **ByteTrack instead of BoT-SORT** — BoT-SORT defaults to sparse optical-flow
   camera-motion compensation, which costs CPU. Embeddings unchanged, but track
   stability and ids change, so replay-test it.

https://docs.ultralytics.com/models/yoloe ·
https://docs.ultralytics.com/integrations/tensorrt ·
https://onnxruntime.ai/docs/performance/tune-performance/threading.html
