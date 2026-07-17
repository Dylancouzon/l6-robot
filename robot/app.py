"""Live demo app: webcam + mic, viewed in the browser; or headless replay.

    uv run python -m robot.app                 # live -> http://127.0.0.1:8765
    uv run python -m robot.app --host 0.0.0.0  # live, reachable from a phone/iPad
    uv run python -m robot.app --source DIR    # headless: replay images/video

The live view is a local web page (OpenCV's macOS windows break on
multi-monitor setups). Controls, in the browser tab (laptop keys) or as
touch buttons (phone/iPad):
             T / hold TEACH  teach the focused object (speak while held)
             A / hold ASK    ask "what did you see today?" by voice
             R / REBOOT      close the shard, reload from disk, re-ask
             F / FORGET      delete what it knows about the recognized object
             Q / IGNORE      dismiss the current unknown (clutter you won't teach)
Quit with Ctrl-C in the terminal (no on-screen quit — a stray tap would end
the demo). On a phone the mic is the phone's own (hold-to-talk, uploaded as a WAV);
--host non-loopback serves HTTPS so the browser will grant mic access.
"""
import argparse
import queue
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from robot import audio, models
from robot.core import Robot
from robot.memory import RECOGNIZE_THRESHOLD

# High contrast, styled for the domain, readable from across a room (BGR).
BG = (246, 244, 240)
INK = (45, 38, 32)
RED = (76, 36, 220)      # Qdrant red
TEAL = (136, 150, 0)
ORANGE = (0, 152, 255)
VIOLET = (255, 71, 96)
PANEL_W = 480
FONT = cv2.FONT_HERSHEY_DUPLEX
PORT = 8765
UTTERANCE_WAV = "/tmp/l6-utterance.wav"  # one buffer; the busy gate serializes writes

PAGE = b"""<!doctype html><title>L6 Robot Memory</title>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<style>
  html,body { margin:0; height:100%; overflow:hidden; background:#0b0b0b;
              font-family:system-ui,sans-serif; touch-action:manipulation }
  #app  { width:100vw; height:100vh; display:flex; flex-direction:column }
  #view { flex:1; min-height:0; width:100%; object-fit:contain; display:block }
  #bar  { display:flex; gap:8px; padding:10px; flex:0 0 auto; background:#e8e5df }
  button { flex:1; min-height:64px; border:0; border-radius:12px; font-size:22px;
           font-weight:700; color:#fff; -webkit-user-select:none; user-select:none;
           touch-action:none }
  #teach  { background:#009688 }
  #ask    { background:#6047ff }
  #reboot { background:#20262d }
  #forget { background:#c0392b }
  #ignore { background:#6b7280 }
  button.rec { box-shadow:0 0 0 4px #fff inset; filter:brightness(1.25) }
  #hint { display:none; position:fixed; top:8px; left:50%;
          transform:translateX(-50%); background:#20262d; color:#fff;
          padding:8px 14px; border-radius:20px; font-size:14px; z-index:9 }
  @media (orientation:portrait) { #hint { display:block } }
</style>
<div id="app">
  <img id="view" src="/stream">
  <div id="bar">
    <button id="teach">HOLD&nbsp;&middot;&nbsp;TEACH</button>
    <button id="ask">HOLD&nbsp;&middot;&nbsp;ASK</button>
    <button id="reboot">REBOOT</button>
    <button id="forget">FORGET</button>
    <button id="ignore">IGNORE</button>
  </div>
</div>
<div id="hint">&#8635; rotate to landscape</div>
<script>
  const $ = id => document.getElementById(id);
  const img = $('view');
  img.onerror = () => setTimeout(() => { img.src = '/stream?' + Date.now(); }, 1000);

  let lock = null;
  const wake = async () => { try { lock = await navigator.wakeLock.request('screen'); } catch (e) {} };
  wake();
  addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') wake(); });

  // Laptop keyboard drives the sounddevice mic on the machine itself.
  addEventListener('keydown', e => {
    const k = e.key.toLowerCase();
    if ('tarfq'.includes(k)) fetch('/key?k=' + k);
  });
  $('reboot').onclick = () => fetch('/key?k=r');
  $('forget').onclick = () => fetch('/key?k=f');
  $('ignore').onclick = () => fetch('/key?k=q');

  // Hold TEACH / ASK to record from THIS device's mic and upload one WAV.
  const WORKLET = URL.createObjectURL(new Blob([
    "class Rec extends AudioWorkletProcessor{process(i){const c=i[0][0];" +
    "if(c)this.port.postMessage(c.slice(0));return true}}" +
    "registerProcessor('rec',Rec)"], {type:'text/javascript'}));
  let ctx, stream, chunks = [], holding = false, rate = 16000;

  async function start(kind, btn) {
    if (holding) return;
    holding = true; chunks = []; btn.classList.add('rec');
    fetch('/listen?k=' + kind);   // narrate "LISTENING" on the filmed view
    try {
      stream = await navigator.mediaDevices.getUserMedia(
        { audio: { echoCancellation: true, noiseSuppression: true } });
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      await ctx.resume();
      await ctx.audioWorklet.addModule(WORKLET);
      rate = ctx.sampleRate;
      const node = new AudioWorkletNode(ctx, 'rec');
      node.port.onmessage = e => chunks.push(e.data);
      ctx.createMediaStreamSource(stream).connect(node);
    } catch (e) {
      holding = false; btn.classList.remove('rec'); alert('mic unavailable: ' + e);
    }
  }
  async function stop(kind, btn) {
    if (!holding) return;
    holding = false; btn.classList.remove('rec');
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (ctx) await ctx.close();
    fetch('/audio?k=' + kind, { method: 'POST', body: encodeWav(chunks, rate) });
  }
  function encodeWav(chunks, rate) {
    const n = chunks.reduce((a, c) => a + c.length, 0);
    const buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
    const s = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
    s(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); s(8, 'WAVE'); s(12, 'fmt ');
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true);
    v.setUint16(32, 2, true); v.setUint16(34, 16, true); s(36, 'data');
    v.setUint32(40, n * 2, true);
    let o = 44;
    for (const c of chunks) for (let i = 0; i < c.length; i++, o += 2) {
      const x = Math.max(-1, Math.min(1, c[i]));
      v.setInt16(o, x < 0 ? x * 0x8000 : x * 0x7fff, true);
    }
    return buf;
  }
  for (const [id, k] of [['teach', 't'], ['ask', 'a']]) {
    const btn = $(id);
    btn.addEventListener('pointerdown', e => { e.preventDefault(); start(k, btn); });
    btn.addEventListener('pointerup',   e => { e.preventDefault(); stop(k, btn); });
    btn.addEventListener('pointercancel', () => stop(k, btn));
  }
</script>
"""


def _lan_ip():
    """LAN IP via a dummy UDP connect (no packets sent). Loopback on failure."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _ensure_cert():
    """Self-signed cert so the phone browser treats the page as a secure
    context — getUserMedia refuses plain http. openssl ships on macOS and
    JetPack; the cert lives in ./cert (gitignored). Returns (cert, key)."""
    root = Path(__file__).resolve().parent.parent / "cert"
    cert, key = root / "cert.pem", root / "key.pem"
    if not (cert.exists() and key.exists()):
        root.mkdir(exist_ok=True)
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
             "-keyout", str(key), "-out", str(cert), "-days", "3650",
             "-subj", "/CN=l6-robot"],
            check=True, capture_output=True)
    return str(cert), str(key)


def _text(img, s, xy, scale=0.8, color=INK, thick=1):
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def _chip(img, s, xy, bg, scale=0.8):
    (w, h), _ = cv2.getTextSize(s, FONT, scale, 2)
    x, y = xy
    cv2.rectangle(img, (x - 6, y - h - 8), (x + w + 6, y + 8), bg, -1)
    _text(img, s, (x, y), scale, (255, 255, 255), 2)


def draw_feed(frame, tracks, focused):
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.box)
        known = t.label is not None
        color = TEAL if known else RED
        thick = 6 if t is focused else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        if t.last_query:
            tag = (f"{t.label}  {t.score:.2f}" if known
                   else f"UNKNOWN  {t.score:.2f}")
            _chip(frame, tag, (x1 + 4, max(30, y1 - 12)), color)
    return frame


def draw_gauge(panel, y, score, threshold):
    """The evidence: live score against the threshold line."""
    x0, x1 = 30, PANEL_W - 30
    lo, hi = 0.5, 1.0
    px = lambda v: int(x0 + (max(lo, min(hi, v)) - lo) / (hi - lo) * (x1 - x0))
    cv2.rectangle(panel, (x0, y), (x1, y + 26), (225, 222, 215), -1)
    if score > lo:
        color = TEAL if score >= threshold else RED
        cv2.rectangle(panel, (x0, y), (px(score), y + 26), color, -1)
    tx = px(threshold)
    cv2.line(panel, (tx, y - 8), (tx, y + 34), INK, 3)
    _text(panel, f"{threshold:.2f}", (tx - 32, y - 14), 0.7, INK, 2)
    _text(panel, f"{score:.3f}", (x1 - 92, y + 62), 1.0, INK, 2)
    _text(panel, "match", (x0, y + 62), 0.7)


def draw_panel(h, events, count, focused, banner, card,
               threshold=RECOGNIZE_THRESHOLD, where=None):
    panel = np.full((h, PANEL_W, 3), BG, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (PANEL_W, 54), VIOLET, -1)
    _text(panel, "ROBOT MEMORY", (20, 37), 1.0, (255, 255, 255), 2)
    _text(panel, f"{count} memories", (PANEL_W - 190, 37),
          0.7, (255, 255, 255), 1)
    if where:
        _text(panel, f"here: {where}", (20, 80), 0.6, VIOLET, 1)
    y = 100
    if focused is None or not focused.last_query:
        _text(panel, "looking...", (30, y), 0.9)
    elif focused.label:
        _chip(panel, focused.label, (30, y), TEAL, 1.0)
        if focused.note:
            for i, line in enumerate(_wrap(f'"{focused.note}"', 38)[:2]):
                _text(panel, line, (30, y + 34 + 26 * i), 0.62)
        y += 34 + 26 * 2
    else:
        _chip(panel, "UNKNOWN — press T to teach", (30, y), RED, 0.85)
        y += 60
    draw_gauge(panel, 180, focused.score if focused else 0.0, threshold)
    y = 290
    if banner:
        cv2.rectangle(panel, (0, y - 30), (PANEL_W, y + 12), ORANGE, -1)
        _text(panel, banner, (20, y), 0.85, (255, 255, 255), 2)
    y = 340
    if card and card[0] == "taught":
        # the shot-2 evidence: one point, both named vectors, the words kept
        taught = card[1]
        cv2.rectangle(panel, (14, y - 24), (PANEL_W - 14, y + 116), TEAL, 3)
        _text(panel, "MEMORY WRITTEN", (26, y), 0.75, TEAL, 2)
        _text(panel, "vectors: image + text", (26, y + 30), 0.7, INK, 2)
        ty = y + 58
        if where:
            _text(panel, f"here: {where}", (26, y + 54), 0.6, VIOLET, 1)
            ty = y + 80
        for i, line in enumerate(_wrap(f'"{taught["transcript"]}"', 40)[:2]):
            _text(panel, line, (26, ty + 24 * i), 0.58)
    elif card and card[0] == "answer":
        q, res = card[1]
        for line in _wrap(f'Q: "{q}"', 40)[:2]:
            _text(panel, line, (20, y), 0.65, VIOLET, 1); y += 24
        _text(panel, "SEEN", (20, y + 6), 0.7, INK, 2); y += 20
        for hit in res["seen"][:2]:
            p = hit.payload
            thumb = cv2.imread(p["thumb"]) if p.get("thumb") else None
            x = 30
            if thumb is not None and y + 56 < h - 40:
                panel[y:y + 56, 24:80] = cv2.resize(thumb, (56, 56))
                x = 92
            when = time.strftime("%H:%M", time.localtime(p["ts"]))
            _text(panel, f"{when}  {p.get('label') or 'unknown'}",
                  (x, y + 22), 0.58)
            tail = f"{p['where']} · " if p.get("where") else ""
            _text(panel, f"{tail}score {hit.score:.2f}", (x, y + 46), 0.5,
                  VIOLET, 1)
            y += 62
        y += 12
        _text(panel, "HEARD", (20, y + 6), 0.7, INK, 2); y += 30
        for hit in res["heard"][:2]:
            p = hit.payload
            what = p.get("transcript") or p.get("label") or "?"
            when = time.strftime("%H:%M", time.localtime(p["ts"]))
            line = _wrap(f"{when}  {what}", 42)[0]
            _text(panel, f"{line}  ({hit.score:.2f})", (30, y), 0.58)
            y += 22
            if p.get("where"):
                _text(panel, f"   {p['where']}", (30, y), 0.5, VIOLET, 1)
                y += 20
    log = events[-3:]
    ly = h - 44 - 22 * len(log)
    if log:
        _text(panel, "memory writes", (20, ly), 0.55, VIOLET, 1)
    for i, e in enumerate(log):
        _text(panel, e[:44], (20, ly + 22 * (i + 1)), 0.55)
    cv2.rectangle(panel, (0, h - 34), (PANEL_W, h), INK, -1)
    _text(panel, "T teach  A ask  R reboot  F forget  Q ignore", (20, h - 11),
          0.65, (255, 255, 255), 1)
    return panel


def _wrap(s, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur] if cur else lines or [""]


class StreamHandler(BaseHTTPRequestHandler):
    app = None  # set by LiveApp

    def log_message(self, *args):
        pass

    def do_GET(self):
        try:
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(PAGE)
            elif self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                while not self.app.stop.is_set():
                    jpeg = self.app.jpeg
                    if jpeg:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                            + jpeg + b"\r\n")
                    time.sleep(0.04)
            elif self.path.startswith("/key?k="):
                self.app.keys.put(self.path[-1])
                self.send_response(204)
                self.end_headers()
            elif self.path.startswith("/listen"):
                # phone pressed hold-to-talk: narrate the beat + grab the crop
                self.app.on_listen("t" if self.path.endswith("=t") else "a")
                self.send_response(204)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass  # tab closed or refreshed mid-stream

    def do_POST(self):
        try:
            if self.path.startswith("/audio"):
                kind = "t" if self.path.endswith("=t") else "a"
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                self.send_response(self.app.on_audio(kind, body))
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass


class LiveApp:
    def __init__(self, robot, camera=0):
        self.robot = robot
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.lock = threading.Lock()   # robot is shared: detect thread + keys
        self.latest = None
        self.tracks = []
        self.banner = None
        self.card = None   # ("taught", {...}) or ("answer", (q, results))
        self.mem_count = 0
        self.jpeg = None   # latest composed view, ready for the stream
        self.keys = queue.Queue()
        self.busy = False  # a voice action is running; ignore T/A meanwhile
        self.focused = None       # what the loop is attending to (panel/phone)
        self.teachable = None     # the salient UNKNOWN — teach target, never a known
        self.pending_crop = None  # crop stashed when the phone starts a teach
        self.stop = threading.Event()

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

    def _render(self, frame, tracks, focused):
        view = draw_feed(frame.copy(), tracks, focused)
        panel = draw_panel(view.shape[0], self.robot.events, self.mem_count,
                           focused, self.banner, self.card,
                           self.robot.memory.threshold, self.robot.memory.where)
        ok, buf = cv2.imencode(".jpg", np.hstack([view, panel]),
                               [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            self.jpeg = buf.tobytes()

    def _voice_action(self, kind, crop):
        """Laptop escape hatch: record via sounddevice, then process. The
        phone path uploads its own WAV and calls _process directly."""
        wav = UTTERANCE_WAV
        self.banner = "LISTENING — speak now"
        try:
            spoke = audio.record_wav(wav)  # stops itself after trailing silence
        except Exception as e:
            print(f"mic failed: {e}")
            self.banner = "mic failed — check MIC_DEVICE in robot/audio.py"
            self.busy = False
            return
        self._process(kind, wav, crop, heard=spoke)

    def _phone_audio(self, kind, body, crop):
        """Phone hold-to-talk: the uploaded WAV replaces the sounddevice
        recording; everything downstream is identical to the laptop mic."""
        wav = UTTERANCE_WAV
        with open(wav, "wb") as f:
            f.write(body)
        self._process(kind, wav, crop)

    def _process(self, kind, wav, crop, heard=None):
        """Silence guard → transcribe → teach/ask, off the main loop so the
        feed never freezes. Shared by both mic paths; clears `busy` when done.

        `heard` is record_wav's speech flag on the laptop path; the phone WAV
        passes None and falls back to the RMS guard."""
        try:
            rms = audio.wav_rms(wav)
            print(f"recorded level (rms): {rms:.0f}"
                  + ("  <- all zeros: no audio reached the robot (mic "
                     "permission?)" if rms == 0 else ""))
            silent = audio.is_silent(wav) if heard is None else not heard
            if silent:
                self.banner = "didn't hear anything — try again"
                return
            self.banner = "thinking..."
            if kind == "t":
                with self.lock:
                    taught = self.robot.teach(crop, wav)
                    self.mem_count = self.robot.memory.count()
                    for t in self.robot.detector.tracks.values():
                        t.last_query = 0  # requery now: watch it recognize
                print(f'taught "{taught["label"]}": {taught["transcript"]!r}')
                self.card = ("taught", taught)
                self.banner = f'taught: "{taught["label"]}"'
            else:
                with self.lock:
                    q, res = self.robot.ask_from_wav(wav)
                print(f"asked: {q!r}")
                self.card = ("answer", (q, res))
                self.banner = None
        finally:
            self._drain_keys()  # drop presses queued while this ran
            self.busy = False

    def on_listen(self, kind):
        """Phone started hold-to-talk. Narrate LISTENING and, for teach,
        stash the crop in focus now — the object may drift before release."""
        self.banner = "LISTENING — speak now"
        if kind == "t":
            f = self.teachable
            self.pending_crop = (f.crop.copy()
                                 if f is not None and f.crop is not None else None)

    def on_audio(self, kind, body):
        """Phone released hold-to-talk with a WAV. Returns an HTTP status.
        Mirrors the main loop's busy-gate. Serialized by the single phone UI
        in the demo; add a lock if multiple clients are ever allowed."""
        if self.busy:
            return 409
        if kind == "t" and self.pending_crop is None:
            self.banner = "nothing in focus to teach"
            return 409
        self.busy = True
        crop = self.pending_crop if kind == "t" else None
        threading.Thread(target=self._phone_audio, args=(kind, body, crop),
                         daemon=True).start()
        return 202

    def run(self, host="127.0.0.1"):
        sys.setswitchinterval(0.002)  # keeps the feed smooth while YOLO runs
        StreamHandler.app = self
        server = ThreadingHTTPServer((host, PORT), StreamHandler)
        loopback = host in ("127.0.0.1", "localhost")
        scheme = "http"
        if not loopback:  # phone mic needs HTTPS (a secure context)
            try:
                cert, key = _ensure_cert()
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(cert, key)
                server.socket = ctx.wrap_socket(server.socket, server_side=True)
                scheme = "https"
            except Exception as e:
                print(f"HTTPS setup failed ({e}); serving HTTP — the phone mic "
                      "will not work, but the stream and REBOOT/quit still do.")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"{scheme}://{host if loopback else _lan_ip()}:{PORT}"
        print(f"live view: {url}  (keys work in the browser tab)")

        def splash(msg):
            print(msg)
            img = np.full((720, 1280 + PANEL_W, 3), BG, np.uint8)
            _text(img, msg, (400, 350), 1.1, INK, 2)
            _text(img, "first run downloads the models (~1.5 GB)",
                  (400, 400), 0.7, VIOLET, 1)
            ok, buf = cv2.imencode(".jpg", img)
            if ok:
                self.jpeg = buf.tobytes()

        splash("warming up...")
        if loopback:
            webbrowser.open(url)  # no browser on the headless robot
        models.warm_up(lambda name: splash(f"loading {name}..."))
        splash("loading YOLOE detector...")
        self.robot.detector.warm()
        self._drain_keys()  # ignore keys pressed before the feed was live
        threading.Thread(target=self._detect_loop, daemon=True).start()
        try:
            self._loop()
        except KeyboardInterrupt:
            pass  # Ctrl-C is the quit path; fall through to a clean shutdown
        self.stop.set()
        server.shutdown()
        self.cap.release()
        self.robot.close()

    def _loop(self):
        """Main thread: pull frames, compose the view, act on key presses."""
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.latest = frame
            tracks = list(self.tracks)
            focused = self.robot.focused(tracks)
            self.focused = focused
            # teach the most salient UNKNOWN, not whatever's focused — a known
            # object could otherwise steal focus and get relabeled
            teachable = self.robot.focused([t for t in tracks if not t.label])
            self.teachable = teachable  # phone teach reads this at hold-start
            self._render(frame, tracks, focused)
            try:
                key = self.keys.get_nowait()
            except queue.Empty:
                key = None
            if key == "t" and not self.busy:
                if teachable is None:
                    self.banner = "nothing new to teach"
                    continue
                self.busy = True
                threading.Thread(target=self._voice_action,
                                 args=("t", teachable.crop.copy()),
                                 daemon=True).start()
            elif key == "a" and not self.busy:
                self.busy = True
                threading.Thread(target=self._voice_action,
                                 args=("a", None),
                                 daemon=True).start()
            elif key == "f":
                # forget the recognized object the panel is showing (focused)
                if focused is None or not focused.label:
                    self.banner = "nothing recognized to forget"
                else:
                    with self.lock:
                        self.robot.forget(focused.label)
                        self.mem_count = self.robot.memory.count()
                        for t in self.robot.detector.tracks.values():
                            t.last_query = 0  # requery: watch it go UNKNOWN
                    self.banner = f'forgot "{focused.label}"'
            elif key == "q":
                # dismiss the current unknown so the robot stops offering it
                if teachable is None:
                    self.banner = "no unknown to ignore"
                else:
                    with self.lock:
                        self.robot.ignore(teachable.tid)
                    self.banner = "ignored — won't track that"
            elif key == "r":
                with self.lock:
                    n = self.robot.reboot()
                    self.mem_count = n
                    if self.card and self.card[0] == "answer":
                        q = self.card[1][0]
                        self.card = ("answer", (q, self.robot.ask(q)))
                self.banner = f"rebooted from disk — {n} memories"

    def _drain_keys(self):
        """Drop key presses queued while a blocking teach/ask ran."""
        while not self.keys.empty():
            self.keys.get_nowait()


def replay(robot, source):
    """Headless file mode: a directory of images, or a video file."""
    src = Path(source)
    if src.is_dir():
        frames = []
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                img = cv2.imread(str(p))
                frames += [(p.name, img)] * 4  # repeats satisfy the stability gate
    else:
        cap, frames, i = cv2.VideoCapture(str(src)), [], 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append((f"frame{i}", f)); i += 1
    now = time.time()
    prev = None
    for name, frame in frames:
        if src.is_dir() and name != prev:
            robot.detector.reset()  # each image is its own scene
            prev = name
        now += 3.0  # spaced past the requery interval
        tracks = robot.process_frame(frame, now)
        f = robot.focused(tracks)
        if f and f.last_query == now:
            verdict = f.label or "UNKNOWN"
            print(f"{name}: {verdict} ({f.score:.3f})")
    robot.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", help="image dir or video file (headless)")
    # resolve against the repo root, not cwd — same as the weights path, so
    # launching from another directory finds the same shard instead of a fresh one
    ap.add_argument("--data",
                    default=str(Path(__file__).resolve().parent.parent / "edge-data"),
                    help="shard directory")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 serves the view on the network (phone/iPad)")
    ap.add_argument("--threshold", type=float, default=RECOGNIZE_THRESHOLD,
                    help="recognition threshold (calibration knob; if it "
                         "moves for the shoot, L5 moves with it)")
    ap.add_argument("--conf", type=float, default=None,
                    help="detector confidence floor (raise to track less)")
    ap.add_argument("--max-area", type=float, default=None,
                    help="biggest proposal kept, as a frame fraction "
                         "(default 0.20 drops torso-sized boxes)")
    ap.add_argument("--location", default=None,
                    help='place stamped on memories this session, e.g. '
                         '"Hotel room" — shown on recall so "where are my keys" '
                         'points back to where it learned them')
    ap.add_argument("--reset", action="store_true",
                    help="wipe the shard dir before starting (clean slate "
                         "between takes; off the live UI so it can't be tapped)")
    args = ap.parse_args()
    if args.reset and Path(args.data).exists():
        shutil.rmtree(args.data)
        print(f"reset: cleared {args.data}")
    robot = Robot(data_dir=args.data, threshold=args.threshold,
                  conf=args.conf, max_area=args.max_area, where=args.location)
    if args.source:
        replay(robot, args.source)
    else:
        LiveApp(robot, args.camera).run(args.host)


if __name__ == "__main__":
    main()
