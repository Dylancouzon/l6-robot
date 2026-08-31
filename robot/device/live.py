"""Camera, browser UI, and worker threads for the live robot.

The main thread renders frames. Separate workers handle detection, ordered
button presses, and voice actions. Code that needs both locks takes the app
lock before the memory lock.
"""
import os
import queue
import shutil
import signal
import ssl
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from robot import config
from robot.brain import models
from robot.brain.labels import norm_label
from robot.device import mic
from robot.device.draw import BG, INK, VIOLET, draw_feed, text
from robot.device.runtime import UTTERANCE_WAV, stamp, when
from robot.device.server import StreamHandler, ensure_cert, lan_ip

PORT = 8765
# `busy` ensures that only one voice action uses the shared audio file.
STREAM_QUALITY = 85
CROP_PX = 180   # the "sees now" thumbnail served at /crop.jpg


def answer_line(res):
    """Build the recall sentence shown in the panel and spoken when possible."""
    s = res["sightings"]
    if res["inventory"]:
        if not s:
            return "I didn't see anything today."
        names = [h.payload.get("label") or "something" for h in s[:3]]
        joined = (", ".join(names[:-1]) + f" and {names[-1]}"
                  if len(names) > 1 else names[0])
        return f"Today I saw {joined}."
    if not res["label"]:
        return "You haven't taught me anything yet."
    if not s:
        return f"I know {res['label']}, but I haven't seen it around."
    p = s[0].payload
    # Teaching and later recognition describe different events.
    verb = "You showed me" if p.get("kind") == "taught" else "I saw"
    line = f"{verb} {res['label']} at {when(p['ts'])}"
    if p.get("where"):
        line += f", in {p['where']}"
    if len(s) > 1:
        line += ". Before that at " + " and ".join(
            when(h.payload["ts"]) for h in s[1:])
    return line + "."


def _hit_json(hit):
    """Convert one time-ordered recall result for the panel."""
    p = hit.payload
    shot = p.get("scene") or p.get("thumb")
    return {
        "when": stamp(p["ts"]),
        "kind": p.get("kind"),
        "label": p.get("label"),
        "where": p.get("where"),
        "thumb": Path(shot).name if shot else None,
    }


def _speak(res):
    """Say the answer out loud on a machine that can. macOS `say`; the Jetson
    has no speaker, and the sentence is on the panel either way."""
    line = answer_line(res)
    if shutil.which("say"):
        subprocess.Popen(["say", line])
    return line


def crop_edges(frame):
    """Remove the configured lens rim from a live camera frame."""
    left, top, right, bottom = config.FRAME_CROP
    if not any(config.FRAME_CROP):
        return frame
    h, w = frame.shape[:2]
    x1, y1 = int(w * left), int(h * top)
    x2, y2 = w - int(w * right), h - int(h * bottom)
    # OpenCV's later encoders expect contiguous image data.
    return frame[y1:y2, x1:x2].copy()


class LiveApp:
    def __init__(self, robot, camera=0, watchdog=0.0):
        self.robot = robot
        self.watchdog = watchdog    # seconds without a frame before exiting
        # DirectShow usually opens Windows webcams faster and respects sizing.
        self.cap = cv2.VideoCapture(
            camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if sys.platform == "win32" and not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(camera, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise SystemExit(
                f"camera {camera} did not open. Try another index with "
                "--camera 1, or let this terminal use the camera: on Windows, "
                "Settings > Privacy & security > Camera; on macOS, System "
                "Settings > Privacy & Security > Camera.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.lock = threading.Lock()   # robot is shared: detect thread + keys
        self.latest = None
        self.tracks = []
        self._banner = None   # the status line; assign through `banner` below
        self._banner_seq = 0
        self.card = None      # ("taught", {...}) or ("answer", (q, results))
        self.mem_count = 0
        self.shot = None      # (seq, jpeg): latest composed view, for /stream
        self._seq = 0
        self.keys = queue.Queue()
        self.busy = False     # a voice action is running; ignore T/A meanwhile
        self.pending_teach = None  # (crop, frame, box) stashed when teach starts
        self.pending_track = None  # ...and the track they came from
        self.stop = threading.Event()
        self.frame_at = None       # monotonic stamp of the last frame

    @property
    def banner(self):
        return self._banner

    @banner.setter
    def banner(self, msg):
        # The sequence lets the page distinguish a repeated status message.
        self._banner = msg
        self._banner_seq += 1

    # -- threads ---------------------------------------------------------------

    def _start_voice(self):
        """Claim the shared voice path if it is idle."""
        with self.lock:
            if self.busy:
                return False
            self.busy = True
            return True

    def _finish_voice(self):
        with self.lock:
            self.busy = False

    def _loop(self):
        """Read and render camera frames on the main thread."""
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("the camera stopped delivering frames; shutting down.")
                break
            if config.CAMERA_ROTATE == 180:
                # Rotate once so detection, memories, and the feed agree.
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = crop_edges(frame)
            self.frame_at = time.monotonic()  # for _watchdog
            self.latest = frame
            self._render(frame, list(self.tracks), self.robot.attention)

    def _detect_loop(self):
        last = 0.0
        while not self.stop.is_set():
            if self.latest is None or time.time() - last < 0.12:
                time.sleep(0.01)
                continue
            frame = self.latest.copy()
            last = time.time()
            with self.lock:
                self.tracks = self.robot.process_frame(frame)
                self.mem_count = self.robot.memory.count()

    def _watchdog(self, stall):
        """Exit after a camera stall so the service manager can restart."""
        self.frame_at = time.monotonic()
        while not self.stop.wait(2):
            if time.monotonic() - self.frame_at > stall:
                print(f"no camera frame for {stall:.0f}s; exiting so the "
                      "service restarts", flush=True)
                # The camera driver may hold the main thread and app lock.
                os._exit(1)

    def _key_loop(self):
        """Key presses and touch buttons, off the frame pump."""
        while not self.stop.is_set():
            try:
                key = self.keys.get(timeout=0.2)  # short, so Ctrl-C is prompt
            except queue.Empty:
                continue
            try:
                self._handle_key(key)
            except Exception:  # noqa: BLE001 - keep the input worker alive
                print(f"key {key!r} failed:")
                traceback.print_exc()

    def _handle_key(self, key):
        focused, teachable = self.robot.attention, self.robot.teachable
        if key == "t":
            target = self._teach_target(teachable)
            if target is None:
                self.banner = "nothing new to teach"
                return
            if not self._start_voice():
                return
            # Requery only the track being taught.
            self.pending_track = teachable
            threading.Thread(target=self._voice_action, args=("t", target),
                             daemon=True).start()
        elif key == "a":
            if not self._start_voice():
                return
            threading.Thread(target=self._voice_action, args=("a", None),
                             daemon=True).start()
        elif key == "f":
            # Destructive focus-based deletion is keyboard-only.
            if focused is None or not focused.label:
                self.banner = "nothing recognized to forget"
            else:
                self.forget_label(focused.label)
        elif key == "q":
            if teachable is None:
                self.banner = "no unknown to ignore"
            else:
                with self.lock:
                    self.robot.ignore(teachable, self.latest)
                    # Replace the shared list so the box disappears immediately.
                    self.tracks = [t for t in self.tracks if t is not teachable]
                self.banner = "ignored · undo it in MEMORY"
        elif key == "r":
            with self.lock:
                n, ms = self.robot.reboot()
                self.mem_count = n
                if self.card and self.card[0] == "answer":
                    q = self.card[1][0]
                    self.card = ("answer", (q, self.robot.ask(q)))
            self.banner = f"rebooted in {ms:.0f} ms · {n} memories"

    def _drain_keys(self):
        """Drop key presses that arrived before the feed went live. Called once,
        before the key thread starts - two consumers on one queue would race."""
        while not self.keys.empty():
            self.keys.get_nowait()

    # -- the view --------------------------------------------------------------

    def _render(self, frame, tracks, focused):
        """The stream carries the annotated camera view and nothing else. The
        panel is HTML, so it lays out for the screen it lands on."""
        view = draw_feed(frame.copy(), tracks, focused)
        ok, buf = cv2.imencode(".jpg", view,
                               [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
        if ok:
            self._publish(buf)

    def _publish(self, buf):
        """Publish a numbered JPEG so each client sends it once."""
        self._seq += 1
        self.shot = (self._seq, buf.tobytes())

    def crop_jpeg(self):
        """The live crop of whatever holds focus: the "sees now" half of the
        match. None when nothing is in focus, which the page reads as a 404."""
        focus = self.robot.attention
        crop = focus.crop if focus is not None else None
        if crop is None or not crop.size:
            return None
        h, w = crop.shape[:2]
        scale = CROP_PX / max(h, w)
        if scale < 1:
            crop = cv2.resize(crop, (max(1, int(w * scale)),
                                     max(1, int(h * scale))))
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    # -- what the page reads ---------------------------------------------------

    def state(self):
        """Return the current panel state as JSON-ready data."""
        r = self.robot
        focus, teachable = r.attention, r.teachable
        seen = focus is not None and focus.last_query
        return {
            "count": self.mem_count,
            "where": r.memory.where,
            "threshold": r.memory.threshold,
            "busy": self.busy,
            "status": self.banner,
            "status_seq": self._banner_seq,
            # Let the button show whether teaching is currently available.
            "teachable": teachable is not None and teachable.crop is not None,
            "focus": {
                "label": focus.label,
                "guess": focus.guess,  # near-miss label, display only
                "score": round(focus.score, 3),
                "note": focus.note,
                "thumb": Path(focus.thumb).name if focus.thumb else None,
            } if seen else None,
            "card": self._card_json(),
            "events": r.events[-3:],
        }

    def memories(self, offset=0, limit=12):
        """Return one page of objects, newest first."""
        objects = self.robot.memory.objects()
        return {
            "total": len(objects),
            "offset": offset,
            "points": self.mem_count,
            "where": self.robot.memory.where,
            "objects": [{
                "label": o["label"],
                "seen": o["seen"],
                "views": [self._view_json(v) for v in o["views"]],
                # Sightings use the same browser shape as taught views.
                "sightings": [self._view_json(v) for v in o["sightings"]],
            } for o in objects[offset:offset + limit]],
        }

    @staticmethod
    def _view_json(payload):
        """One taught view or sighting, flattened for the page."""
        thumb = payload.get("thumb")
        # Older points may have only the masked crop.
        scene = payload.get("scene") or thumb
        return {
            # Large point IDs must remain exact in JavaScript.
            "id": str(payload["id"]),
            "when": stamp(payload.get("ts") or 0),
            "transcript": payload.get("transcript"),
            "where": payload.get("where"),
            "thumb": Path(thumb).name if thumb else None,
            "scene": Path(scene).name if scene else None,
        }

    def ignored(self):
        """Everything IGNORE has dismissed, for the tab to list and undo. Read
        from the shard, not the detector: dismissals outlive the session."""
        rows = self.robot.memory.ignored()
        return {"ignored": [{
            "pid": str(r.id),
            "when": stamp(r.payload.get("ts") or 0),
            "thumb": (Path(r.payload["thumb"]).name
                      if r.payload.get("thumb") else None),
        } for r in rows]}

    def _card_json(self):
        """The last action's result: taught, forgot, or an answer."""
        card = self.card
        if not card:
            return None
        kind, data = card
        if kind == "taught":
            return {"kind": "taught", "label": data["label"],
                    "transcript": data["transcript"],
                    "where": self.robot.memory.where}
        if kind == "forgot":
            label, n = data
            return {"kind": "forgot", "label": label, "n": n}
        q, res = data
        return {"kind": "answer", "q": q, "say": answer_line(res),
                "hits": [_hit_json(h) for h in res["sightings"][:3]]}

    # -- what the page changes -------------------------------------------------

    def forget_label(self, label):
        """Delete one object by name - from the tab, or from the F key."""
        with self.lock:
            n = self.robot.forget(label)
            self.mem_count = self.robot.memory.count()
        self.card = ("forgot", (label, n))
        self.banner = f'forgot "{label}"'
        return n

    def rename(self, label, to):
        """Rename an object and update the vector used for text recall."""
        to = norm_label(to)
        models.warm_text()
        text_vec = models.embed_text(to)
        with self.lock:
            n = self.robot.rename(label, to, text_vec=text_vec)
        self.banner = f'renamed to "{to}"'
        return n

    def forget_view(self, pid, label):
        """Drop one taught view, or one sighting occasion, from the tab."""
        with self.lock:
            n, whole = self.robot.forget_view(pid, label)
            self.mem_count = self.robot.memory.count()
        if whole:
            self.card = ("forgot", (label, n))
            self.banner = f'that was the last view of "{label}"; forgot it'
        elif n > 1:
            self.banner = f'dropped {n} photos of "{label}"'
        elif n:
            self.banner = f'dropped one view of "{label}"'
        else:
            self.banner = "that view is already gone"
        return {"n": n, "whole": whole, "label": label}

    def unignore(self, pid):
        with self.lock:
            ok = self.robot.unignore(pid)
        self.banner = "tracking that again" if ok else "that isn't ignored"
        return ok

    def confirm(self, label):
        """Teach the current crop using its displayed near-match label."""
        self.banner = "thinking..."
        models.warm_encoders()
        with self.lock:
            res = self.robot.confirm(label, frame=self.latest)
            if res:
                self.mem_count = self.robot.memory.count()
        if res is None:
            # The focused guess changed after the page rendered.
            self.banner = "nothing to confirm"
            return {"ok": False}
        self.card = ("taught", res)
        self.banner = f'taught: "{res["label"]}"'
        return {"ok": True, "label": res["label"]}

    def set_where(self, place):
        """Update the location stamped on future memories."""
        with self.lock:
            where = self.robot.set_where(place)
        self.banner = f'here: "{where}"' if where else "location cleared"
        return where

    # -- the voice path --------------------------------------------------------

    def _teach_target(self, track):
        """Everything a teach needs from the live view, grabbed in one go.

        Copied, not referenced: the detect thread keeps rewriting a track's
        crop, and the object may have moved by the time the finger lifts.
        """
        if track is None or track.crop is None:
            return None
        frame = self.latest
        return (track.crop.copy(),
                None if frame is None else frame.copy(),
                track.box)

    def on_listen(self, action):
        """Phone started hold-to-talk: narrate LISTENING, stash the crop."""
        self.banner = "LISTENING · speak now"
        # Load voice models while the button is held to shorten the later wait.
        threading.Thread(target=self._warm_quietly, daemon=True).start()
        if action == "t":
            f = self.robot.teachable
            self.pending_teach = self._teach_target(f)
            self.pending_track = f if self.pending_teach else None

    def on_audio(self, action, body):
        """Accept a phone recording and return an HTTP status."""
        if not self._start_voice():
            return 409
        if action == "t" and self.pending_teach is None:
            self.banner = "nothing in focus to teach"
            self._finish_voice()
            return 409
        target = self.pending_teach if action == "t" else None
        threading.Thread(target=self._phone_audio, args=(action, body, target),
                         daemon=True).start()
        return 202

    def _warm_quietly(self):
        """Load voice models in the background; the action retries failures."""
        try:
            models.warm_encoders()
        except Exception as e:  # noqa: BLE001 - the action retries the load
            print(f"encoder warm failed (the action will retry): {e}")

    def _voice_action(self, action, target):
        """Record from the computer's microphone, then process the audio."""
        self.banner = "LISTENING · speak now"
        try:
            spoke = mic.record_wav(UTTERANCE_WAV)  # stops itself after silence
        except Exception as e:  # noqa: BLE001 - device APIs vary by platform
            print(f"mic failed: {e}")
            self.banner = "mic failed: check MIC_DEVICE in .env"
            self._finish_voice()
            return
        self._process(action, UTTERANCE_WAV, target, heard=spoke)

    def _phone_audio(self, action, body, target):
        """Phone path: the uploaded WAV replaces the local recording, and
        everything downstream is identical."""
        try:
            with open(UTTERANCE_WAV, "wb") as f:
                f.write(body)
        except OSError as e:
            print(f"could not save phone audio: {e}")
            self.banner = "could not save the recording"
            self._finish_voice()
            return
        self._process(action, UTTERANCE_WAV, target)

    def _process(self, action, wav, target, heard=None):
        """Reject silence, transcribe, then teach or recall."""
        try:
            rms = mic.wav_rms(wav)
            print(f"recorded level (rms): {rms:.0f}"
                  + ("  <- all zeros: no audio reached the robot (mic "
                     "permission?)" if rms == 0 else ""))
            silent = mic.is_silent(wav) if heard is None else not heard
            if silent:
                self.banner = "didn't hear anything, try again"
                return
            self.banner = "thinking..."
            # Trimming silence speeds Whisper and reduces hallucinations.
            kept = mic.trim_to_speech(wav)
            if kept:
                print(f"trimmed the hold down to {kept:.1f} s of speech")
            # Model loading and transcription stay outside the live-state lock.
            models.warm_encoders()
            q = models.transcribe(wav)
            if action == "t":
                crop, frame, box = target
                with self.lock:
                    taught = self.robot.teach(crop, q, frame=frame, box=box)
                    self.mem_count = self.robot.memory.count()
                    if self.pending_track is not None:
                        # Refresh only the track that supplied the taught crop.
                        self.pending_track.requery_now()
                print(f'taught "{taught["label"]}": {taught["transcript"]!r}')
                self.card = ("taught", taught)
                self.banner = f'taught: "{taught["label"]}"'
            else:
                # Recall uses Memory's lock and does not touch track state.
                res = self.robot.ask(q)
                print(f"asked: {q!r}")
                self.card = ("answer", (q, res))
                self.banner = None
                _speak(res)
        finally:
            self._finish_voice()

    # -- startup and shutdown --------------------------------------------------

    def run(self, host="127.0.0.1", advertise=None):
        sys.setswitchinterval(0.002)  # keeps the feed smooth while YOLO runs
        StreamHandler.app = self
        server = ThreadingHTTPServer((host, PORT), StreamHandler)
        # The listening host and the address advertised to phones can differ.
        loopback = host in ("127.0.0.1", "localhost")
        addr = advertise or (host if loopback else lan_ip())
        scheme = "http"
        if not loopback:  # the phone mic needs HTTPS (a secure context)
            try:
                cert, key = ensure_cert(addr)
                StreamHandler.cert = cert  # offered at /cert.crt
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(cert, key)
                server.socket = ctx.wrap_socket(server.socket, server_side=True)
                scheme = "https"
            except Exception as e:  # noqa: BLE001 - HTTP remains a useful fallback
                print(f"HTTPS setup failed ({e}); serving HTTP, so the phone "
                      "mic will not work, but the stream and REBOOT/quit do.")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"{scheme}://{addr}:{PORT}"
        print(f"live view: {url}  (keys work in the browser tab)")
        if scheme == "https":
            print("the browser will warn once (nothing vouches for a "
                  "self-signed cert): accept it and the mic will work.\n"
                  f"to silence it for good on your demo phone, open {url}"
                  "/cert.crt and trust it - see docs/phone.md.")

        try:
            self._splash("warming up...")
            if loopback:
                webbrowser.open(url)  # no browser on the headless robot
            models.warm_up(lambda name: self._splash(f"loading {name}..."))
            self._splash("loading YOLOE detector...")
            self.robot.detector.warm()
            self._drain_keys()  # ignore keys pressed before the feed was live
            threading.Thread(target=self._detect_loop, daemon=True).start()
            threading.Thread(target=self._key_loop, daemon=True).start()
            if self.watchdog:
                threading.Thread(target=self._watchdog, args=(self.watchdog,),
                                 daemon=True).start()
            self._loop()
        except KeyboardInterrupt:
            pass  # Ctrl-C is the quit path; fall through to a clean shutdown
        finally:
            # Ignore repeated interrupts while native resources close.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            self.stop.set()
            server.shutdown()
            self.cap.release()
            with self.lock:       # let an in-flight write finish: closing the
                self.robot.close()  # shard under one can corrupt it

    def _splash(self, msg):
        """A still frame on the stream while the models load, so a headless
        robot shows progress instead of a dead feed."""
        print(msg)
        img = np.full((720, 1280, 3), BG, np.uint8)
        text(img, msg, (380, 350), 1.1, INK, 2)
        text(img, "first run downloads the models (~1.5 GB)",
             (380, 400), 0.7, VIOLET, 1)
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            self._publish(buf)
