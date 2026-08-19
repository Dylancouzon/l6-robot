"""The live app: camera, threads, buttons, and what the panel says.

Four threads, and the split between them is the whole performance story:

    main        pull frames, draw boxes, encode the JPEG. Takes NO lock.
    detect      run the detector and match memory. Holds the live-state lock
                for a whole pass, which is why nothing else may wait on it.
    keys        one thread, so two taps on a button run in order.
    voice       one per action, so the feed never freezes while Whisper thinks.

Lock order is always the app's live-state lock, then the memory lock. Never
the reverse.
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

from robot.brain import models
from robot.device import mic
from robot.device.draw import BG, INK, VIOLET, draw_feed, text
from robot.device.server import StreamHandler, ensure_cert, lan_ip

PORT = 8765
UTTERANCE_WAV = "/tmp/l6-utterance.wav"  # one buffer; `busy` serializes writes
# JPEG quality for the streamed feed. Turn it down only against a measurement
# of your own scene - JPEG size depends far more on what the camera sees.
STREAM_QUALITY = 85
CROP_PX = 180   # the "sees now" thumbnail served at /crop.jpg


def _when(ts):
    """A spoken timestamp. Today keeps just the clock; older sightings name
    the day, because "I saw it at 9:12 PM" is a lie by omission on Tuesday."""
    t = time.localtime(ts)
    if time.strftime("%Y%m%d", t) == time.strftime("%Y%m%d"):
        return time.strftime("%-I:%M %p", t)
    return time.strftime("%b %-d, %-I:%M %p", t)


def answer_line(res):
    """The sentence recall answers with. Every value in it is a live payload.

    Built separately from speaking it, because the panel shows it too: the
    Jetson has no audio hardware, so on the appliance this line IS the answer.
    """
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
    line = f"I saw {res['label']} at {_when(p['ts'])}"
    if p.get("where"):
        line += f", in {p['where']}"
    if len(s) > 1:
        line += ". Before that at " + " and ".join(
            _when(h.payload["ts"]) for h in s[1:])
    return line + "."


def _hit_json(hit):
    """One sighting row for the panel. No score: these come from a scroll
    ordered by time, not a vector search."""
    p = hit.payload
    return {
        "when": time.strftime("%b %-d, %H:%M", time.localtime(p["ts"])),
        "label": p.get("label"),
        "where": p.get("where"),
        "thumb": Path(p["thumb"]).name if p.get("thumb") else None,
    }


def _speak(res):
    """Say the answer out loud on a machine that can. macOS `say`; the Jetson
    has no speaker, and the sentence is on the panel either way."""
    line = answer_line(res)
    if shutil.which("say"):
        subprocess.Popen(["say", line])
    return line


class LiveApp:
    def __init__(self, robot, camera=0, watchdog=0.0):
        self.robot = robot
        self.watchdog = watchdog    # seconds without a frame before exiting
        self.cap = cv2.VideoCapture(camera)
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
        # A property only so each assignment bumps a counter: the page hides a
        # status line a few seconds after it arrives, and without one it cannot
        # tell a stale message from the same message sent again.
        self._banner = msg
        self._banner_seq += 1

    # -- threads ---------------------------------------------------------------

    def _loop(self):
        """Main thread: pull frames and compose the view, nothing else.

        Keep it that way. FORGET scans and flushes the shard and REBOOT
        reopens it, and both want the lock the detect thread holds for a whole
        detection pass - doing that here is a visible stall on the button.
        """
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
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
        """Insurance for a robot nobody can see: if frames stop arriving, die
        so the service manager restarts us.

        A camera that fails cleanly already ends _loop. This is for a USB
        device that wedges INSIDE the driver call, which the frame pump cannot
        notice because it is the thread that is stuck. The clock is armed here
        rather than on the first frame, because a camera that wedges on its
        very first read is exactly what this exists for.

        Monotonic on purpose: the operator sets the appliance's date by hand,
        and a stepped wall clock would either kill a healthy robot or blind
        this entirely.
        """
        self.frame_at = time.monotonic()
        # wait, not sleep, so a shutdown in progress cannot be shot from under
        while not self.stop.wait(2):
            if time.monotonic() - self.frame_at > stall:
                print(f"no camera frame for {stall:.0f}s — exiting so the "
                      "service restarts", flush=True)
                # _exit, not a clean shutdown: the pump is blocked in a driver
                # call and self.lock may never come free. Safe because every
                # memory write is already flushed to disk.
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
            except Exception:  # a stray press must not kill the thread
                print(f"key {key!r} failed:")
                traceback.print_exc()

    def _handle_key(self, key):
        focused, teachable = self.robot.attention, self.robot.teachable
        if key == "t" and not self.busy:
            target = self._teach_target(teachable)
            if target is None:
                self.banner = "nothing new to teach"
                return
            self.busy = True
            # remember which track this taught, so only that one re-asks memory
            self.pending_track = teachable
            threading.Thread(target=self._voice_action, args=("t", target),
                             daemon=True).start()
        elif key == "a" and not self.busy:
            self.busy = True
            threading.Thread(target=self._voice_action, args=("a", None),
                             daemon=True).start()
        elif key == "f":
            # Forget the recognized object the panel is showing. Keyboard only:
            # aiming a delete with the camera is what the memory tab replaces,
            # but a key is still the fastest beat to film and is not something
            # a thumb lands on by accident.
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
                    # and out of the list the frame pump draws from, so the box
                    # goes on the next composed frame rather than at the next
                    # detect pass. A new list, not an edit: the pump reads this
                    # without the lock.
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
        """Hand a composed JPEG to the stream handler, stamped so it can send
        each frame exactly once. One tuple, assigned in one bytecode, so the
        number and the bytes can never disagree."""
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
        """What the panel draws, as JSON. Read once from the shared attributes
        so a detect pass landing mid-build cannot split one view across two."""
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
            # whether TEACH will do anything, so the button can say so before
            # the finger goes down instead of refusing afterwards
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
        """One page of "what do you know?", newest object first.

        No app lock, deliberately: Memory serializes its own shard access, and
        behind the app lock this page waited on a whole detection pass - paid
        on every open, delete and rename, since all of them redraw the tab.
        """
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
                # the robot's own photos of it, same shape as a view. The
                # card's tap-cycle walks these after the taught views.
                "sightings": [self._view_json(v) for v in o["sightings"]],
            } for o in objects[offset:offset + limit]],
        }

    @staticmethod
    def _view_json(payload):
        """One taught view or sighting, flattened for the page."""
        thumb = payload.get("thumb")
        # the unmasked picture, falling back to the crop so points written
        # before scenes existed still show something
        scene = payload.get("scene") or thumb
        return {
            # A string on purpose. Point ids are around 1.8e18, past
            # JavaScript's 2^53, where JSON numbers silently round - the page
            # would then ask to delete an id that does not exist and the view
            # would quietly survive its own deletion.
            "id": str(payload["id"]),
            "when": time.strftime("%b %-d, %H:%M",
                                  time.localtime(payload.get("ts") or 0)),
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
            "when": time.strftime("%b %-d, %H:%M",
                                  time.localtime(r.payload.get("ts") or 0)),
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
        """Rename one object, from the tab. No vector moves."""
        with self.lock:
            n = self.robot.rename(label, to)
        self.banner = f'renamed to "{to}"'
        return n

    def forget_view(self, pid, label):
        """Drop one taught view, or one sighting occasion, from the tab."""
        with self.lock:
            n, whole = self.robot.forget_view(pid, label)
            self.mem_count = self.robot.memory.count()
        if whole:
            # it was the last view, so the object itself is gone - say so with
            # the same card FORGET uses, or the tab looks like it over-deleted
            self.card = ("forgot", (label, n))
            self.banner = f'that was the last view of "{label}" — forgot it'
        elif n > 1:
            # a sighting row stands for its burst, so one tap drops several
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
        """Tap on the orange guess: teach the attending crop as that name.

        Encoders warm OUTSIDE the lock, exactly like the voice path: paying a
        first-of-session model load under the lock stalls the detect thread
        long enough to start killing tracks.
        """
        self.banner = "thinking..."
        models.warm_encoders()
        with self.lock:
            res = self.robot.confirm(label, frame=self.latest)
            if res:
                self.mem_count = self.robot.memory.count()
        if res is None:
            # the guess moved on between the paint and the tap: refuse honestly
            # rather than teach whatever holds the panel now
            self.banner = "nothing to confirm"
            return {"ok": False}
        self.card = ("taught", res)
        self.banner = f'taught: "{res["label"]}"'
        return {"ok": True, "label": res["label"]}

    def set_where(self, place):
        """Update the robot's location, from the tab. Under the lock like the
        other mutations: a teach mid-flight reads this and should see one
        value, not a torn decision."""
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
        # Start loading Whisper and Nomic NOW, under the hold, so the first
        # voice action of a session does not pay ~7 s of model loads after the
        # finger lifts. The cost is moved, not removed: a cold load pauses
        # every thread including the feed, so it happens while "LISTENING" is
        # on screen. Safe for tracking because a ~3 s hold sits inside
        # DEAD_SECONDS, and that margin is not large.
        threading.Thread(target=self._warm_quietly, daemon=True).start()
        if action == "t":
            f = self.robot.teachable
            self.pending_teach = self._teach_target(f)
            self.pending_track = f if self.pending_teach else None

    def on_audio(self, action, body):
        """Phone released hold-to-talk with a WAV. Returns an HTTP status.

        ponytail: the check-and-set on `busy` is not atomic. One operator, and
        shard writes are serialized by the lock regardless.
        """
        if self.busy:
            return 409
        if action == "t" and self.pending_teach is None:
            self.banner = "nothing in focus to teach"
            return 409
        self.busy = True
        target = self.pending_teach if action == "t" else None
        threading.Thread(target=self._phone_audio, args=(action, body, target),
                         daemon=True).start()
        return 202

    def _warm_quietly(self):
        """warm_encoders for a background thread: a failed load must be one log
        line, not a traceback storm - the action itself retries it."""
        try:
            models.warm_encoders()
        except Exception as e:
            print(f"encoder warm failed (the action will retry): {e}")

    def _voice_action(self, action, target):
        """Laptop path: record through the local mic, then process.

        No early encoder warm here, unlike on_listen: a cold model load blocks
        this thread's 100 ms reads and chops the audio. The phone records on
        the phone, so it has nothing to starve.
        """
        self.banner = "LISTENING · speak now"
        try:
            spoke = mic.record_wav(UTTERANCE_WAV)  # stops itself after silence
        except Exception as e:
            print(f"mic failed: {e}")
            self.banner = "mic failed: check MIC_DEVICE in robot/device/mic.py"
            self.busy = False
            return
        self._process(action, UTTERANCE_WAV, target, heard=spoke)

    def _phone_audio(self, action, body, target):
        """Phone path: the uploaded WAV replaces the local recording, and
        everything downstream is identical."""
        with open(UTTERANCE_WAV, "wb") as f:
            f.write(body)
        self._process(action, UTTERANCE_WAV, target)

    def _process(self, action, wav, target, heard=None):
        """Silence guard -> transcribe -> teach or ask. Shared by both mic
        paths, off the main loop so the feed never freezes.

        `heard` is record_wav's speech flag on the laptop path; the phone WAV
        passes None and falls back to the RMS guard.
        """
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
            # Hand Whisper the utterance, not the hold: silence on either end
            # both slows it down and makes it hallucinate.
            kept = mic.trim_to_speech(wav)
            if kept:
                print(f"trimmed the hold down to {kept:.1f} s of speech")
            # Before the transcribe, so Whisper is certainly loaded, and before
            # the live-state lock, so a first-of-session load cannot freeze the
            # detect thread long enough to start killing tracks. The warm that
            # started at pointerdown may still be running; this waits on its
            # lock rather than building the same model twice.
            models.warm_encoders()
            # Whisper takes seconds; run it before claiming the lock so the
            # detector keeps tracking while the robot listens.
            q = models.transcribe(wav)
            if action == "t":
                crop, frame, box = target
                with self.lock:
                    taught = self.robot.teach(crop, q, frame=frame, box=box)
                    self.mem_count = self.robot.memory.count()
                    if self.pending_track is not None:
                        # the beat: watch that one box turn green. One embed,
                        # not one per track - a burst of them under the lock is
                        # what the detect thread stalls on.
                        self.pending_track.requery_now()
                print(f'taught "{taught["label"]}": {taught["transcript"]!r}')
                self.card = ("taught", taught)
                self.banner = f'taught: "{taught["label"]}"'
            else:
                # No app lock: ask reads only memory, which serializes its own
                # shard access, and touches no track state.
                res = self.robot.ask(q)
                print(f"asked: {q!r}")
                self.card = ("answer", (q, res))
                self.banner = None
                _speak(res)
        finally:
            # No key drain here: the key thread handles presses as they arrive,
            # and racing it for the queue could throw before `busy` is cleared,
            # which wedges teach and ask until restart.
            self.busy = False

    # -- startup and shutdown --------------------------------------------------

    def run(self, host="127.0.0.1", advertise=None):
        sys.setswitchinterval(0.002)  # keeps the feed smooth while YOLO runs
        StreamHandler.app = self
        server = ThreadingHTTPServer((host, PORT), StreamHandler)
        # `host` says where to listen; the URL and the certificate need one
        # real address to name. Normally that is whichever one the LAN gave us,
        # but the headless robot passes --advertise: it serves its own hotspot
        # at a fixed address, and naming that address is what lets a phone
        # trust the certificate once and never be asked again.
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
            except Exception as e:
                print(f"HTTPS setup failed ({e}); serving HTTP, so the phone "
                      "mic will not work, but the stream and REBOOT/quit do.")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"{scheme}://{addr}:{PORT}"
        print(f"live view: {url}  (keys work in the browser tab)")
        if scheme == "https":
            print("the browser will warn once (nothing vouches for a "
                  "self-signed cert): accept it and the mic will work.\n"
                  f"to silence it for good on your demo phone, open {url}"
                  "/cert.crt and trust it - see the README.")

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
            # From here the interrupt has been heard, and a SECOND one must not
            # land in the middle of this: it would skip the shard close and
            # abort in a native destructor. Two arrive as a matter of course -
            # a service manager signals every process in the cgroup, and a
            # human holding Ctrl-C does the same.
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
