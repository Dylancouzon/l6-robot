# Code and Comment Readability Feedback

## Verdict

The code is understandable, but it is not yet as clean and readable as it can be. The core architecture is easy to follow: `app.py` owns the live UI, `core.py` owns the robot loop, `detect.py` owns tracking, `memory.py` owns Qdrant Edge, and `models.py` owns model loading. That separation is good.

The main readability problems are drift between docs and code, comments that carry production shorthand, and `robot/app.py` doing too many jobs in one large file.

## Highest-Impact Fixes

### 1. Bring README controls back in sync with the app

`README.md:45-50` still says `Q` quits. The current app uses:

- `F` / **FORGET** to delete the focused recognized object
- `Q` / **IGNORE** to dismiss the current unknown
- Ctrl-C in the terminal to quit

The app docstring and on-screen footer are current, but the README is stale. This is the most visible readability issue because a user will trust the README first.

Relevant code: `robot/app.py:10-17`, `robot/app.py:76-80`, `robot/app.py:307`.

### 2. Split or tame `robot/app.py`

`robot/app.py` is readable in chunks, but at ~685 lines it now contains:

- CLI entrypoint
- HTML/CSS/JS page
- HTTP request handler
- camera loop
- detection thread orchestration
- voice workflow
- OpenCV drawing code
- replay mode

That is a lot of modes in one place. The embedded `PAGE` string at `robot/app.py:51-158` is the biggest readability drag because HTML, CSS, JS, and Python all share one file and one indentation style.

Suggested split:

- `robot/ui_page.py` or `robot/web.py`: `PAGE`, request parsing helpers
- `robot/draw.py`: `_text`, `_chip`, `draw_feed`, `draw_gauge`, `draw_panel`, `_wrap`
- keep `LiveApp`, `replay`, and `main` in `app.py`

If you want to avoid new files, at least move `PAGE` below the drawing helpers or load it from a small template file.

### 3. Remove private shorthand from comments

Several comments still sound like notes from a production plan rather than durable source comments. They are vivid, but they make the code feel less clean to a future reader.

Examples to rewrite:

- `robot/app.py:258`: "shot-2 evidence"
- `robot/app.py:354`: "narrate the beat"
- `robot/app.py:659`: "moves for the shoot"
- `robot/app.py:671`: "between takes"
- `robot/core.py:131`: "the reboot beat"
- `robot/memory.py:79`: "offline-reboot beat"
- `robot/memory.py:94-95`: "power-cycles the device"
- `robot/memory.py:102` and `robot/memory.py:126`: "ponytail"

Suggested style: explain the invariant, not the filming context. For example:

```python
# Flush immediately so a restart demonstrates persistence.
```

instead of:

```python
# durable now: the offline-reboot beat power-cycles the device...
```

### 4. Keep the good explanatory comments

Some comments are doing real work and should stay:

- `robot/detect.py:21-27`: explains the macOS MPS autorelease pool workaround.
- `robot/detect.py:130-132`: explains repo-root weight resolution.
- `robot/app.py:173-176`: explains HTTPS for mobile mic access.
- `robot/audio.py:75-81`: explains silence handling and Whisper hallucination risk.
- `robot/memory.py:38-43`: explains why recall dedupes by label.
- `robot/memory.py:167-168`: explains the two recall spaces.

These comments answer "why," not "what," which is the right bar.

### 5. Make key names less surprising

Using `Q` for ignore is surprising because `Q` usually means quit, and the previous README still reflects that. If the demo can tolerate it, `I` for ignore would be clearer.

If `Q` must stay for muscle memory, make that explicit everywhere:

- app docstring
- README controls table
- on-screen footer
- phone button label

Right now the code is consistent internally, but the product language is not.

### 6. Use URL parsing in `StreamHandler`

`robot/app.py:349-355` and `robot/app.py:366-367` infer the key from `self.path[-1]` or `endswith("=t")`. That works for the current tiny API, but it is harder to read and easy to break if a query parameter is added.

Suggested cleanup:

```python
from urllib.parse import parse_qs, urlparse

params = parse_qs(urlparse(self.path).query)
key = params.get("k", [""])[0]
```

This would make `/key`, `/listen`, and `/audio` clearer.

### 7. The `busy` race comment should not wave away the race

`robot/app.py:487-490` says the check-then-set race is harmless for a single presenter. That is probably true for the demo, but comments that dismiss a race tend to age poorly.

Cleaner options:

- Guard `busy` with `self.lock` or a dedicated `threading.Lock`.
- Or reword the comment to state the scope: "This is serialized by the single phone UI in the demo; use a lock if multiple clients are allowed."

### 8. Avoid comments that mirror code too closely

Some comments mostly restate the next line:

- `robot/app.py:475`: `_drain_keys()` already says it drains keys.
- `robot/app.py:538`: similar.
- `robot/app.py:582-593`: the branches and banner strings already explain forget/ignore.
- `robot/detect.py:212`: `crop_quality` and `crop_q` already make the crop-selection idea visible.

These are not harmful, but removing a few would make the remaining comments feel more intentional.

### 9. Clarify course-coupling comments

The repo is course-specific, so references to L2-L6 are okay in docs. In code, use them only where they enforce a real compatibility contract.

Good candidate to keep:

- `robot/models.py:1-5`, because it explains why the model stack must stay aligned.

Good candidates to soften:

- `robot/detect.py:1-12`
- `robot/memory.py:26-27`
- `robot/core.py:107`
- `robot/memory.py:121-123`

Example rewrite:

```python
# Keep this default aligned with the lesson threshold unless the lesson changes too.
```

### 10. Clean up small readability rough edges

- `robot/core.py:26-28`: `if v` drops valid falsey values like `0.0`. Prefer `if v is not None`; it is both clearer and safer.
- `robot/app.py:312-319`: `_wrap` uses a semicolon on line 316. Split it into two lines for normal Python shape.
- `robot/memory.py:59`: the inline `where` comment is long. Move it above the assignment or shorten it.
- `robot/audio.py:9-11`: `MIC_DEVICE` setup is useful, but the comment is a mini instruction block. Consider moving that detail to README and leaving one short code comment.
- `pyproject.toml:12`: the inline dependency comment is long. It is useful, but it makes the dependency list harder to scan; consider moving it above the line.

## Module Notes

### `robot/app.py`

This is the file most in need of cleanup. The behavior is understandable, but the reader has to jump between page JS, HTTP endpoints, UI rendering, and thread state. Extracting drawing and page content would make the main application flow much easier to review.

The strongest comments explain platform constraints: browser UI instead of OpenCV windows, HTTPS for phone mic access, and Ctrl-C as the quit path. The weakest comments are filming-specific phrases and comments that narrate obvious branches.

### `robot/core.py`

This file is in good shape. It reads like the domain model: process frame, teach, forget, ignore, ask, reboot. The section dividers are helpful.

The main cleanup is tone: replace "beat" and lesson shorthand with neutral descriptions unless the comment is explicitly about course parity.

### `robot/detect.py`

The detector comments are mostly strong because they explain calibration, stability, person suppression, and platform behavior. This file has the best "why" comments in the repo.

The top docstring could be less course-scripted. The `memory-fleet` provenance notes are useful, but they should either be concise or moved to README/BOM-level context.

### `robot/memory.py`

The data model is clean and readable. The named vectors and payload fields are easy to trace.

The biggest issue is tone. "ponytail" should be removed. The flush comment should be kept but made less cinematic. The L2/L5 references should be softened unless they are actively enforcing course parity.

### `robot/audio.py`

This file is compact and practical. The silence guard comment is useful and should stay. `parse_label` is simple enough to understand.

The module docstring's ffmpeg note is helpful if this file is still being actively debugged. If the repo is meant to be polished, move that history into a short comment near `record_wav` or drop it.

### `robot/models.py`

Very readable. The model functions are small and well named. The course-alignment docstring is appropriate here because this module is the contract for "same stack as the lesson."

### `testdata/verify_scores.py`

Readable for a scratch check. If this is meant to stay in the repo, rename "Scratch check" to "Fixture check" or "Threshold check" so it feels intentional.

## Suggested Cleanup Order

1. Fix the README controls table for `F`, `Q`, and Ctrl-C.
2. Replace the production shorthand comments: "beat," "shoot," "takes," "shot-2," and "ponytail."
3. Split `robot/app.py` drawing helpers into a small `robot/draw.py`.
4. Split or externalize the browser `PAGE`.
5. Replace path suffix parsing with real query parsing.
6. Tighten the falsey-option check in `Robot.__init__`.
7. Do a final pass deleting comments that restate nearby code.

## Bottom Line

The code is close. It already has a clear mental model and many good comments. The cleanup needed now is mostly editorial: make comments durable, remove private shorthand, reduce the size of `app.py`, and keep the user-facing docs synchronized with the current controls.
