"""Live demo app: webcam + mic, viewed in the browser; or headless replay.

    uv run python -m robot.app                 # live -> http://127.0.0.1:8765
    uv run python -m robot.app --host 0.0.0.0  # live, reachable from a phone/iPad
    uv run python -m robot.app --source DIR    # headless: replay images/video

The live view is a local web page (OpenCV's macOS windows break on
multi-monitor setups). Controls, in the browser tab (laptop keys) or as
touch buttons (phone/iPad):
             T / hold TEACH  teach the focused object (speak while held)
             A / hold ASK    ask "what did you see today?" by voice (answers out loud on macOS)
             R / REBOOT      close the shard, reload from disk, re-ask
             F / FORGET      delete what it knows about the recognized object
             Q / IGNORE      dismiss the current unknown (clutter you won't teach)
Quit with Ctrl-C in the terminal (no on-screen quit — a stray tap would end
the demo). On a phone the mic is the phone's own (hold-to-talk, uploaded as a WAV);
--host non-loopback serves HTTPS so the browser will grant mic access.

In its case the robot runs this as a service on its own Wi-Fi, with no keyboard
or screen at all: see deploy/ and the README section "Headless Appliance".
"""
import argparse
import ipaddress
import os
import queue
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from functools import lru_cache
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


def _ensure_cert(ip):
    """Self-signed cert so the phone browser treats the page as a secure
    context — getUserMedia refuses plain http. openssl ships on macOS and
    JetPack; the cert lives in ./cert (gitignored). Returns (cert, key).

    Two details decide how loudly the browser complains, and both are easy to
    get wrong:

    * It must carry a **subjectAltName** for the address you actually open.
      Browsers stopped reading the CN field in 2017, so a CN-only cert is not
      merely untrusted, it names nothing — which is a harsher warning, and
      cannot be fixed by trusting the cert. Hence the SAN list below.
    * Validity must stay under 825 days or Safari refuses it outright, trusted
      or not. 397 days keeps us inside the stricter modern limit too.

    `CA:TRUE` is deliberate: it lets a phone install this as a trusted root
    once and stop warning for good (see the README). Regenerated whenever the
    address changes, because a cert for the old IP names the wrong machine.
    """
    root = Path(__file__).resolve().parent.parent / "cert"
    cert, key, named = root / "cert.pem", root / "key.pem", root / "names.txt"
    # localhost and 127.0.0.1 stay valid so the laptop view works off the same
    # cert; the .local name is for whoever has mDNS working
    want = ",".join([f"IP:{ip}", "IP:127.0.0.1", "DNS:localhost",
                     f"DNS:{socket.gethostname().split('.')[0]}.local"])
    if (cert.exists() and key.exists() and named.exists()
            and named.read_text() == want):
        return str(cert), str(key)
    root.mkdir(exist_ok=True)
    # a config file, not -addext: the latter is missing from the LibreSSL that
    # ships as /usr/bin/openssl on macOS
    conf = root / "openssl.cnf"
    conf.write_text(
        "[req]\nprompt=no\ndistinguished_name=dn\nx509_extensions=ext\n"
        "[dn]\nCN=l6-robot\n"
        f"[ext]\nbasicConstraints=critical,CA:TRUE\nsubjectAltName={want}\n")
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(key), "-out", str(cert), "-days", "397",
         "-config", str(conf)],
        check=True, capture_output=True)
    named.write_text(want)
    print(f"generated a certificate for {ip} (valid 397 days)")
    return str(cert), str(key)


@lru_cache(maxsize=32)
def _thumb_img(path):
    """Thumbnails, read once. The panel is redrawn for every streamed frame,
    and decoding the same JPEGs off disk ten times a second is visible jank on
    the Jetson. Thumbs are write-once (`Robot._thumb` stamps the filename with
    time_ns and `forget` deletes points, not files), so this cannot go stale.
    32 crops at demo resolution is a few MB — keep it bounded, memory on this
    board is the scarce thing."""
    return cv2.imread(path)


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


def draw_gauge(panel, y, score, threshold, x0=30, x1=PANEL_W - 30):
    """The evidence: live score against the threshold line."""
    lo, hi = 0.5, 1.0
    px = lambda v: int(x0 + (max(lo, min(hi, v)) - lo) / (hi - lo) * (x1 - x0))
    cv2.rectangle(panel, (x0, y), (x1, y + 26), (225, 222, 215), -1)
    if score > lo:
        color = TEAL if score >= threshold else RED
        cv2.rectangle(panel, (x0, y), (px(score), y + 26), color, -1)
    tx = px(threshold)
    cv2.line(panel, (tx, y - 8), (tx, y + 34), INK, 3)
    _text(panel, f"{threshold:.2f}", (tx - 32, y - 14), 0.7, INK, 2)
    if score > lo:
        s = f"{score:.3f}"
        (w, _), _ = cv2.getTextSize(s, FONT, 0.8, 2)
        _text(panel, s, ((x0 + x1 - w) // 2, y + 58), 0.8, INK, 2)


def draw_match(panel, y, focused, threshold):
    """The match, visible: the remembered view next to the live view, with
    the score gauge between them — vector similarity an audience can see."""
    left = (_thumb_img(focused.thumb)
            if focused is not None and focused.thumb else None)
    right = focused.crop if focused is not None else None
    for cap, img, x in (("remembers", left, 30),
                        ("sees now", right, PANEL_W - 102)):
        cv2.rectangle(panel, (x, y), (x + 72, y + 72), (225, 222, 215), -1)
        if img is not None and img.size:
            panel[y:y + 72, x:x + 72] = cv2.resize(img, (72, 72))
        (w, _), _ = cv2.getTextSize(cap, FONT, 0.5, 1)
        _text(panel, cap, (x + 36 - w // 2, y + 90), 0.5, VIOLET, 1)
    draw_gauge(panel, y + 20, focused.score if focused else 0.0, threshold,
               x0=118, x1=PANEL_W - 118)


def draw_panel(h, events, count, focused, banner, card,
               threshold=RECOGNIZE_THRESHOLD, where=None):
    panel = np.full((h, PANEL_W, 3), BG, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (PANEL_W, 54), VIOLET, -1)
    if where:
        _text(panel, "ROBOT MEMORY", (20, 25), 0.8, (255, 255, 255), 2)
        _text(panel, f"here: {where}", (20, 46), 0.55, (255, 255, 255), 1)
    else:
        _text(panel, "ROBOT MEMORY", (20, 37), 1.0, (255, 255, 255), 2)
    _text(panel, f"{count} memories", (PANEL_W - 190, 37),
          0.7, (255, 255, 255), 1)
    y = 104
    if focused is None or not focused.last_query:
        _text(panel, "looking...", (30, y), 0.9)
    elif focused.label:
        _chip(panel, focused.label, (30, y), TEAL, 1.0)
    else:
        _chip(panel, "UNKNOWN · press T to teach", (30, y), RED, 0.85)
    draw_match(panel, 128, focused, threshold)
    if focused is not None and focused.note:
        for i, line in enumerate(_wrap(f'"{focused.note}"', 44)[:2]):
            _text(panel, line, (30, 248 + 24 * i), 0.6)
    y = 310
    if banner:
        cv2.rectangle(panel, (0, y - 30), (PANEL_W, y + 12), ORANGE, -1)
        _text(panel, banner, (20, y), 0.85, (255, 255, 255), 2)
    y = 350
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
    elif card and card[0] == "forgot":
        # the correction beat: teach wrong -> forget -> watch it go UNKNOWN
        label, n = card[1]
        cv2.rectangle(panel, (14, y - 24), (PANEL_W - 14, y + 66), RED, 3)
        _text(panel, "MEMORY DELETED", (26, y), 0.75, RED, 2)
        _text(panel, f'"{label[:22]}" · {n} point(s) removed', (26, y + 30),
              0.62, INK, 2)
    elif card and card[0] == "answer":
        q, res = card[1]
        for line in _wrap(f'Q: "{q}"', 40)[:2]:
            _text(panel, line, (20, y), 0.65, VIOLET, 1); y += 24
        _text(panel, "SEEN", (20, y + 6), 0.7, INK, 2); y += 20
        for hit in res["seen"][:2]:
            p = hit.payload
            thumb = _thumb_img(p["thumb"]) if p.get("thumb") else None
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
            if y > h - 130:  # stop before the memory-writes log
                break
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


def _speak(q, res):
    """Say the answer out loud — every spoken value is a live payload field.
    macOS `say`; the Jetson port swaps in espeak."""
    seen = res["seen"]
    if not seen:
        line = "I didn't see anything like that today."
    elif "what did you see" in q.lower():
        # the day-inventory question lists the objects; anything else answers
        # with the top hit. ponytail: one phrase check, not intent parsing —
        # the demo script asks this exact question
        labels = [h.payload.get("label") or "something" for h in seen[:3]]
        names = (", ".join(labels[:-1]) + f" and {labels[-1]}"
                 if len(labels) > 1 else labels[0])
        line = f"Today I saw {names}."
    else:
        p = seen[0].payload
        when = time.strftime("%-I:%M %p", time.localtime(p["ts"]))
        line = f"I saw {p.get('label') or 'something'} at {when}"
        if p.get("where"):
            line += f", in {p['where']}"
        line += "."
    if shutil.which("say"):
        subprocess.Popen(["say", line])
    return line


def _wrap(s, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur] if cur else lines or [""]


class StreamHandler(BaseHTTPRequestHandler):
    app = None   # set by LiveApp
    cert = None  # path to the served cert, when running HTTPS

    def log_message(self, *args):
        pass

    def do_GET(self):
        try:
            if self.path.startswith("/cert.crt") and self.cert:
                # Hand the cert to the phone so it can be trusted once and stop
                # warning. A headless robot has no other easy way to deliver a
                # file, and this is the same cert already on the wire — serving
                # it publicly gives away nothing the TLS handshake doesn't.
                self.send_response(200)
                self.send_header("Content-Type", "application/x-x509-ca-cert")
                self.send_header("Content-Disposition",
                                 'attachment; filename="l6-robot.crt"')
                self.end_headers()
                self.wfile.write(Path(self.cert).read_bytes())
            elif self.path == "/":
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
        except (BrokenPipeError, ConnectionResetError, ssl.SSLEOFError):
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
        except (BrokenPipeError, ConnectionResetError, ssl.SSLEOFError):
            pass


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
        self.banner = None
        self.card = None   # ("taught", {...}) or ("answer", (q, results))
        self.mem_count = 0
        self.jpeg = None   # latest composed view, ready for the stream
        self.keys = queue.Queue()
        self.busy = False  # a voice action is running; ignore T/A meanwhile
        # What the robot attends to lives on the robot (see Robot.attention),
        # decided by the detect thread that owns track state.
        self.pending_crop = None   # crop stashed when a teach starts
        self.pending_track = None  # ...and the track it came from
        self.stop = threading.Event()
        self.frame_at = None       # monotonic stamp of the last frame (_watchdog)

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
        self.banner = "LISTENING · speak now"
        try:
            spoke = audio.record_wav(wav)  # stops itself after trailing silence
        except Exception as e:
            print(f"mic failed: {e}")
            self.banner = "mic failed: check MIC_DEVICE in robot/audio.py"
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
                self.banner = "didn't hear anything, try again"
                return
            self.banner = "thinking..."
            # Whisper takes seconds; run it before claiming the lock so the
            # detector keeps tracking while the robot listens.
            q = models.transcribe(wav)
            if kind == "t":
                with self.lock:
                    taught = self.robot.teach(crop, q)
                    self.mem_count = self.robot.memory.count()
                    if self.pending_track is not None:
                        # the beat: watch that one box turn green. One embed,
                        # not one per track — a burst of them under the lock is
                        # what the detect thread stalls on
                        self.pending_track.requery_now()
                print(f'taught "{taught["label"]}": {taught["transcript"]!r}')
                self.card = ("taught", taught)
                self.banner = f'taught: "{taught["label"]}"'
            else:
                with self.lock:
                    res = self.robot.ask(q)
                print(f"asked: {q!r}")
                self.card = ("answer", (q, res))
                self.banner = None
                _speak(q, res)
        finally:
            # no key drain: the key thread handles presses as they arrive, so
            # nothing piled up here, and racing it for the queue could throw
            # before `busy` is cleared and wedge the robot
            self.busy = False

    def on_listen(self, kind):
        """Phone started hold-to-talk. Narrate LISTENING and, for teach,
        stash the crop in focus now — the object may drift before release."""
        self.banner = "LISTENING · speak now"
        if kind == "t":
            f = self.robot.teachable
            got = f is not None and f.crop is not None
            self.pending_crop = f.crop.copy() if got else None
            self.pending_track = f if got else None

    def on_audio(self, kind, body):
        """Phone released hold-to-talk with a WAV. Returns an HTTP status.
        Mirrors the busy-gate in `_handle_key`. Serialized by the single phone
        UI in the demo; add a lock if multiple clients are ever allowed — the
        check-and-set on `busy` below is not atomic."""
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

    def run(self, host="127.0.0.1", advertise=None):
        sys.setswitchinterval(0.002)  # keeps the feed smooth while YOLO runs
        StreamHandler.app = self
        server = ThreadingHTTPServer((host, PORT), StreamHandler)
        # `host` says where to listen; the URL and the certificate need one real
        # address to name. Normally we look up whichever one the LAN gave us,
        # but the headless robot passes --advertise: it serves its own hotspot
        # at a fixed address, and naming that address (rather than a DHCP lease)
        # is what lets a phone trust the cert once and never be asked again.
        loopback = host in ("127.0.0.1", "localhost")
        addr = advertise or (host if loopback else _lan_ip())
        scheme = "http"
        if not loopback:  # phone mic needs HTTPS (a secure context)
            try:
                cert, key = _ensure_cert(addr)
                StreamHandler.cert = cert  # offered at /cert.crt, see do_GET
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(cert, key)
                server.socket = ctx.wrap_socket(server.socket, server_side=True)
                scheme = "https"
            except Exception as e:
                print(f"HTTPS setup failed ({e}); serving HTTP, so the phone mic "
                      "will not work, but the stream and REBOOT/quit still do.")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"{scheme}://{addr}:{PORT}"
        print(f"live view: {url}  (keys work in the browser tab)")
        if scheme == "https":
            # Nobody vouches for a self-signed cert, so the first visit always
            # warns. Say so here rather than letting it look like a failure.
            print("the browser will warn once (nothing vouches for a "
                  "self-signed cert): accept it and the mic will work.\n"
                  f"to silence it for good on your demo phone, open {url}"
                  "/cert.crt and trust it — see the README.")

        def splash(msg):
            print(msg)
            img = np.full((720, 1280 + PANEL_W, 3), BG, np.uint8)
            _text(img, msg, (400, 350), 1.1, INK, 2)
            _text(img, "first run downloads the models (~1.5 GB)",
                  (400, 400), 0.7, VIOLET, 1)
            ok, buf = cv2.imencode(".jpg", img)
            if ok:
                self.jpeg = buf.tobytes()

        try:
            splash("warming up...")
            if loopback:
                webbrowser.open(url)  # no browser on the headless robot
            models.warm_up(lambda name: splash(f"loading {name}..."))
            splash("loading YOLOE detector...")
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
            # From here on the interrupt has been heard, and a *second* one must
            # not land in the middle of this: it skips the shard close and the
            # process aborts in a native destructor instead. Two arrive as a
            # matter of course — a service manager signals every process in the
            # cgroup, so `uv` forwards one and the kernel delivers another — and
            # a human holding Ctrl-C does the same.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            self.stop.set()
            server.shutdown()
            self.cap.release()
            with self.lock:  # let an in-flight FORGET/REBOOT/teach finish —
                self.robot.close()  # closing under a write can corrupt the shard

    def _loop(self):
        """Main thread: pull frames and compose the view, nothing else.

        Keep it that way. FORGET scans and flushes the shard and REBOOT
        reopens it, and both want the lock the detect thread holds for a whole
        YOLO pass — doing that here is a visible stall on the button, so key
        presses are handled on their own thread.
        """
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.frame_at = time.monotonic()  # for _watchdog; see why monotonic
            self.latest = frame
            self._render(frame, list(self.tracks), self.robot.attention)

    def _watchdog(self, stall):
        """Insurance for a robot nobody can see: if frames stop arriving, die
        so the service manager restarts us.

        A camera that fails cleanly already ends `_loop` — `cap.read()` returns
        False. This is for the other case, a USB device that wedges *inside*
        the driver call and never returns, which the frame pump cannot notice
        because it is the thread that is stuck. Off unless --watchdog asks for
        it, because on a laptop a lid-close looks exactly like a stall.

        Arm the clock here rather than waiting for a first frame to stamp it.
        A camera that wedges on its *very first* read is the likeliest wedge of
        all — it is what a power-cut USB device does — and waiting for a frame
        before arming would leave exactly that case invisible forever.

        The clock is monotonic on purpose. The operator is told to set the
        appliance's date by hand (README, "The Clock"), and a wall clock that
        jumps a day forward would read as a stall and kill a healthy robot.
        """
        self.frame_at = time.monotonic()
        # wait, not sleep: stop.set() ends this thread within 2 s, so a
        # shutdown in progress can't be shot from under itself
        while not self.stop.wait(2):
            if time.monotonic() - self.frame_at > stall:
                print(f"no camera frame for {stall:.0f}s — exiting so the "
                      "service restarts", flush=True)
                # _exit, not a clean shutdown: the pump is blocked in a driver
                # call and self.lock may never come free. Safe because every
                # memory write is already flushed to disk (Memory._upsert).
                os._exit(1)

    def _key_loop(self):
        """Key presses and touch buttons, off the frame pump. One thread, so
        two fast taps on the same button run in order rather than at once."""
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
            if teachable is None or teachable.crop is None:
                self.banner = "nothing new to teach"
                return
            self.busy = True
            # remember which track this taught, so only that one re-asks
            # memory afterwards instead of every box on screen
            self.pending_track = teachable
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
                label = focused.label
                with self.lock:
                    n = self.robot.forget(label)
                    self.mem_count = self.robot.memory.count()
                self.card = ("forgot", (label, n))
                self.banner = f'forgot "{label}"'
        elif key == "q":
            # dismiss the current unknown so the robot stops offering it
            if teachable is None:
                self.banner = "no unknown to ignore"
            else:
                with self.lock:
                    self.robot.ignore(teachable.tid)
                self.banner = "ignored, won't track that"
        elif key == "r":
            with self.lock:
                n, ms = self.robot.reboot()
                self.mem_count = n
                if self.card and self.card[0] == "answer":
                    q = self.card[1][0]
                    self.card = ("answer", (q, self.robot.ask(q)))
            self.banner = f"rebooted in {ms:.0f} ms · {n} memories"

    def _drain_keys(self):
        """Drop key presses that arrived before the feed went live."""
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
        robot.process_frame(frame, now)
        f = robot.attention
        if f and f.last_query == now:
            verdict = f.label or "UNKNOWN"
            print(f"{name}: {verdict} ({f.score:.3f})")
    robot.close()


def _ip_arg(text):
    """--advertise, checked here rather than by openssl.

    `_ensure_cert` writes the value as an `IP:` subjectAltName, and a hostname
    in that field makes openssl reject the whole certificate. That lands in the
    HTTPS fallback and quietly serves plain http, which takes the phone mic away
    with one line of warning — a bad way to find out about a typo.
    """
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not an IP address. The certificate needs an IP here; "
            "for the robot's own hotspot that is 10.42.0.1.")


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
    ap.add_argument("--advertise", default=None, metavar="ADDR", type=_ip_arg,
                    help="IP address to put in the URL and the certificate, "
                         "when it isn't the one to look up (the headless "
                         "robot's own hotspot: a fixed address the phone can "
                         "trust for good)")
    ap.add_argument("--watchdog", type=float, default=0.0, metavar="SECONDS",
                    help="exit if the camera delivers no frame for this long, "
                         "so a service manager can restart the robot; 0 is off "
                         "(unattended runs only — a laptop lid-close looks "
                         "like a stall)")
    # These three are per-camera. Their persistent home is .env (see
    # .env.example and robot/config.py); the flags override it for one run.
    ap.add_argument("--threshold", type=float, default=RECOGNIZE_THRESHOLD,
                    help=f"recognition threshold (.env RECOGNIZE_THRESHOLD, "
                         f"currently {RECOGNIZE_THRESHOLD}); calibrate with "
                         f"testdata/verify_scores.py")
    ap.add_argument("--conf", type=float, default=None,
                    help="detector confidence floor, raise to track less "
                         "(.env DETECT_CONF)")
    ap.add_argument("--max-area", type=float, default=None,
                    help="biggest proposal kept, as a frame fraction; the "
                         "default drops torso-sized boxes (.env "
                         "DETECT_MAX_AREA)")
    ap.add_argument("--location", default=None,
                    help='place stamped on memories this session, e.g. '
                         '"Hotel room", shown on recall so "where are my keys" '
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
        LiveApp(robot, args.camera, watchdog=args.watchdog).run(
            args.host, advertise=args.advertise)


if __name__ == "__main__":
    main()
