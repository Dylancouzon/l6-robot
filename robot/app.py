"""Live demo app: webcam + mic, viewed in the browser; or headless replay.

    uv run python -m robot.app                 # live -> http://127.0.0.1:8765
    uv run python -m robot.app --host 0.0.0.0  # live, reachable from a phone/iPad
    uv run python -m robot.app --source DIR    # headless: replay images/video

The live view is a local web page (OpenCV's macOS windows break on
multi-monitor setups). The page streams the annotated camera feed and polls
`/state` for everything else, so the panel is HTML rather than pixels drawn
into the video — which is what lets it lay out for a phone. Controls, in the
browser tab (laptop keys) or as touch buttons (phone/iPad):
             T / hold TEACH  teach the focused object (speak while held)
             A / hold ASK    ask "what did you see today?" by voice (answers out loud on macOS)
             F / FORGET      delete what it knows about the recognized object
             Q / IGNORE      dismiss the current unknown (clutter you won't teach)
             R               close the shard, reload from disk, re-ask (keyboard
                             only: an offline robot makes the point by itself)
Quit with Ctrl-C in the terminal (no on-screen quit — a stray tap would end
the demo). On a phone the mic is the phone's own (hold-to-talk, uploaded as a WAV);
--host non-loopback serves HTTPS so the browser will grant mic access.

In its case the robot runs this as a service on its own Wi-Fi, with no keyboard
or screen at all: see deploy/ and the README section "Headless Appliance".
"""
import argparse
import ipaddress
import json
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np

from robot import audio, models
from robot.core import Robot
from robot.memory import RECOGNIZE_THRESHOLD

# High contrast, styled for the domain, readable from across a room (BGR).
# Only the boxes drawn onto the feed use these now; the panel is CSS, and the
# same palette is repeated at the top of PAGE as custom properties.
BG = (246, 244, 240)
INK = (45, 38, 32)
RED = (76, 36, 220)      # Qdrant red
TEAL = (136, 150, 0)
VIOLET = (255, 71, 96)
FONT = cv2.FONT_HERSHEY_DUPLEX
PORT = 8765
UTTERANCE_WAV = "/tmp/l6-utterance.wav"  # one buffer; the busy gate serializes writes
# JPEG quality for the streamed feed. Turn it down only if the feed lags on a
# congested channel, and measure first — see "Streaming over the hotspot".
STREAM_QUALITY = 85
CROP_PX = 180   # the "sees now" thumbnail served at /crop.jpg

# ASCII only in here, comments included: a bytes literal cannot hold anything
# else. Use HTML entities in the markup and \\uXXXX escapes in the JS.
PAGE = b"""<!doctype html><title>L6 Robot Memory</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
  /* Same palette as the boxes drawn on the feed (see the BGR constants). */
  :root { --bg:#0b0b0b; --panel:#f0f4f6; --ink:#20262d; --teal:#009688;
          --violet:#6047ff; --red:#dc244c; --dim:#6b7280; --line:#d9e0e4 }
  * { box-sizing:border-box }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--ink);
              font-family:system-ui,-apple-system,sans-serif;
              touch-action:manipulation }
  #app  { height:100%; display:flex; flex-direction:column }
  /* The cap is load-bearing, not taste. The feed is the only element with an
     intrinsic size, and #side is flex:1 (basis 0), so it contributes nothing
     to the used height and cannot push the feed back: a full-width 16:9 image
     eats a landscape phone's entire viewport on its own and the panel computes
     to zero. Reset it in the row layout below, where height is not contended. */
  #view { width:100%; background:#000; object-fit:contain;
          flex:0 1 auto; min-height:0; max-height:50vh }
  #side { flex:1; min-height:0; display:flex; flex-direction:column;
          background:var(--panel) }
  #panel { flex:1; min-height:0; overflow-y:auto; padding:14px 16px }
  /* Side by side wherever there is width for it, which includes a phone in
     landscape (667 CSS px and up), not just a laptop. Portrait keeps the
     column and stays readable, which is why there is no "rotate your phone"
     nag any more. */
  @media (min-width:640px) and (orientation:landscape) {
    #app  { flex-direction:row }
    #view { flex:1; min-width:0; height:100%; max-height:none }
    #side { flex:0 0 min(400px, 45%) }
  }
  #head { display:flex; justify-content:space-between; align-items:baseline;
          border-bottom:2px solid var(--line); padding-bottom:8px }
  h1 { font-size:15px; letter-spacing:.14em; margin:0; color:var(--violet) }
  #count { font-size:14px; color:var(--dim) }
  #where { font-size:13px; color:var(--dim); margin-top:6px }
  #label { display:inline-block; margin:14px 0 12px; padding:6px 14px;
           border-radius:10px; color:#fff; background:var(--dim);
           font-size:clamp(20px,4.5vw,30px); font-weight:700; line-height:1.15;
           word-break:break-word }
  #match { display:flex; align-items:center; gap:12px }
  figure { margin:0; text-align:center; flex:0 0 auto }
  figure img { width:76px; height:76px; object-fit:cover; border-radius:8px;
               background:var(--line); display:block; opacity:0;
               transition:opacity .2s }
  figcaption { font-size:11px; color:var(--violet); margin-top:5px }
  #gauge { flex:1; min-width:0 }
  #track { position:relative; height:22px; border-radius:5px;
           background:var(--line); overflow:hidden }
  #fill { height:100%; width:0; background:var(--red); transition:width .2s }
  #mark { position:absolute; top:0; bottom:0; width:3px; background:var(--ink) }
  #nums { display:flex; justify-content:space-between; margin-top:5px;
          font-size:12px; color:var(--dim) }
  #score { font-weight:700; color:var(--ink); font-size:15px }
  #note { font-size:14px; color:var(--dim); font-style:italic; margin-top:12px }
  .card { border:2px solid var(--dim); border-radius:10px; padding:10px 12px;
          margin-top:16px }
  .card h2 { font-size:13px; letter-spacing:.1em; margin:0 0 8px }
  .row { display:flex; gap:8px; align-items:center; margin-top:8px;
         font-size:14px }
  .row img { width:46px; height:46px; object-fit:cover; border-radius:6px;
             background:var(--line) }
  .meta { font-size:12px; color:var(--violet) }
  #log { margin-top:18px; font-size:12px; color:var(--dim); line-height:1.7 }
  /* Reserve the row permanently and toggle visibility, never display. Toggling
     display made the button bar jump ~37 px on every action and back again when
     the message faded, which reads as a hitch on the one control the operator is
     watching. line-height is pinned so the row's height is exactly 9+20+9=38 px
     and cannot lose to a UA's font metrics by a pixel; nowrap keeps a long
     transcript from wrapping to two lines and reintroducing the shift.
     visibility:hidden hides the dark background too, so an empty row simply
     reads as panel padding. */
  #status { flex:0 0 auto; visibility:hidden; height:38px; padding:9px 16px;
            background:var(--ink); color:#fff; font-size:14px; line-height:20px;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis }
  #bar { flex:0 0 auto; display:grid; grid-template-columns:1fr 1fr; gap:8px;
         padding:10px; background:#e2e8ea;
         padding-bottom:calc(10px + env(safe-area-inset-bottom)) }
  button { border:0; border-radius:12px; color:#fff; font-weight:700;
           -webkit-user-select:none; user-select:none; touch-action:none }
  #teach,#ask { min-height:66px; font-size:clamp(17px,4vw,21px) }
  #teach { background:var(--teal) }
  #ask { background:var(--violet) }
  #forget,#ignore { min-height:44px; font-size:14px; background:var(--dim) }
  #forget { background:var(--red) }
  #teach.off { opacity:.4 }
  button.rec { box-shadow:0 0 0 4px #fff inset; filter:brightness(1.25) }
</style>
<div id="app">
  <img id="view" src="/stream">
  <div id="side">
    <div id="panel">
      <div id="head"><h1>ROBOT MEMORY</h1><span id="count"></span></div>
      <div id="where"></div>
      <div id="label">looking&hellip;</div>
      <div id="match">
        <figure><img id="thumbA"><figcaption>remembers</figcaption></figure>
        <div id="gauge">
          <div id="track"><div id="fill"></div><div id="mark"></div></div>
          <div id="nums"><span id="score"></span><span id="thr"></span></div>
        </div>
        <figure><img id="thumbB"><figcaption>sees now</figcaption></figure>
      </div>
      <div id="note"></div>
      <div id="card"></div>
      <div id="log"></div>
    </div>
    <div id="status"></div>
    <div id="bar">
      <button id="teach">HOLD&nbsp;&middot;&nbsp;TEACH</button>
      <button id="ask">HOLD&nbsp;&middot;&nbsp;ASK</button>
      <button id="forget">FORGET</button>
      <button id="ignore">IGNORE</button>
    </div>
  </div>
</div>
<script>
  const $ = id => document.getElementById(id);
  const img = $('view');
  img.onerror = () => setTimeout(() => { img.src = '/stream?' + Date.now(); }, 1000);

  let lock = null;
  const wake = async () => { try { lock = await navigator.wakeLock.request('screen'); } catch (e) {} };
  wake();
  addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') wake(); });

  // Laptop keyboard drives the sounddevice mic on the machine itself. R (close
  // the shard and reload it from disk) is keyboard-only on purpose: it is a
  // presenter's beat, not a control anyone should meet on a phone.
  addEventListener('keydown', e => {
    const k = e.key.toLowerCase();
    if ('tarfq'.includes(k)) fetch('/key?k=' + k);
  });
  $('forget').onclick = () => fetch('/key?k=f');
  $('ignore').onclick = () => fetch('/key?k=q');

  // ---- the panel: /state, four times a second -----------------------------
  // Everything below used to be drawn into the video with cv2.putText, at a
  // fixed 480 px, which on a phone in portrait rendered about 100 px wide.
  // Text nodes cost nothing on the wire and lay out for the screen they land
  // on. Never build these with innerHTML: labels and notes are whatever
  // Whisper heard.
  const el = (tag, text, cls) => {
    const n = document.createElement(tag);
    if (text != null) n.textContent = text;
    if (cls) n.className = cls;
    return n;
  };
  // The gauge starts at 0.5, not 0: unrelated crops from one camera share
  // lighting and scale and rarely score below it, so the interesting range is
  // the top half. See "Calibrating For Your Camera" in the README.
  const pct = v => Math.max(0, Math.min(1, (v - 0.5) / 0.5)) * 100;
  // Fade a thumbnail out rather than clearing src, and only re-fetch when the
  // url actually changes: thumb filenames are stamped, so a re-set would be a
  // request per poll for a picture the browser already has.
  const setImg = (n, url) => {
    n.style.opacity = url ? 1 : 0;
    if (url && n.dataset.url !== url) { n.dataset.url = url; n.src = url; }
  };

  let shownSeq = null, statusAt = 0, shownCard = '';
  function render(s) {
    $('count').textContent = s.count + ' memories';
    $('where').textContent = s.where ? 'here: ' + s.where : '';
    const f = s.focus;
    const known = f && f.label;
    $('label').textContent = f ? (f.label || 'UNKNOWN \\u2014 hold TEACH')
                               : 'looking\\u2026';
    $('label').style.background =
      f ? (known ? 'var(--teal)' : 'var(--red)') : 'var(--dim)';
    $('fill').style.width = (f ? pct(f.score) : 0) + '%';
    $('fill').style.background =
      f && f.score >= s.threshold ? 'var(--teal)' : 'var(--red)';
    $('mark').style.left = pct(s.threshold) + '%';
    $('score').textContent = f && f.score ? f.score.toFixed(3) : '';
    $('thr').textContent = 'bar ' + s.threshold.toFixed(2);
    setImg($('thumbA'), f && f.thumb ? '/thumb?f=' + f.thumb : null);
    $('note').textContent = f && f.note ? '\\u201c' + f.note + '\\u201d' : '';
    // Say whether TEACH will do anything before the finger goes down, rather
    // than answering "nothing new to teach" after it.
    $('teach').classList.toggle('off', !s.teachable);

    const key = JSON.stringify(s.card);
    if (key !== shownCard) { shownCard = key; drawCard(s.card); }

    const log = $('log');
    log.replaceChildren();
    if (s.events.length) log.append(el('div', 'memory writes'));
    for (const e of s.events) log.append(el('div', e));

    // One status line for every message the robot has: the busy phase, and
    // the refusals ("nothing new to teach") that used to be painted into the
    // video where nobody looks. Fades once it has been read.
    const msg = s.status || (s.busy ? 'working\\u2026' : '');
    // On the counter, not the text: the same message sent twice is two
    // messages. statusAt stays 0 on the first paint so a reload does not
    // replay whatever the robot last said.
    if (s.status_seq !== shownSeq) {
      statusAt = shownSeq === null ? 0 : Date.now();
      shownSeq = s.status_seq;
    }
    const show = msg && (s.busy || Date.now() - statusAt < 6000);
    // visibility, not display: the row is always in the layout (see the CSS)
    // so showing a message cannot move the buttons under a waiting finger.
    $('status').style.visibility = show ? 'visible' : 'hidden';
    $('status').textContent = msg;
  }

  function drawCard(c) {
    const box = $('card');
    box.replaceChildren();
    if (!c) return;
    const card = el('div', null, 'card');
    if (c.kind === 'taught') {
      card.style.borderColor = 'var(--teal)';
      const h = el('h2', 'MEMORY WRITTEN'); h.style.color = 'var(--teal)';
      card.append(h, el('div', 'vectors: image + text'));
      if (c.where) card.append(el('div', 'here: ' + c.where, 'meta'));
      card.append(el('div', '\\u201c' + c.transcript + '\\u201d'));
    } else if (c.kind === 'forgot') {
      card.style.borderColor = 'var(--red)';
      const h = el('h2', 'MEMORY DELETED'); h.style.color = 'var(--red)';
      card.append(h, el('div', '"' + c.label + '" \\u00b7 ' + c.n + ' point(s) removed'));
    } else {
      card.style.borderColor = 'var(--violet)';
      const h = el('h2', 'RECALL'); h.style.color = 'var(--violet)';
      card.append(h, el('div', 'Q: \\u201c' + c.q + '\\u201d'));
      card.append(el('div', c.say, 'meta'));
      for (const [title, hits] of [['SEEN', c.seen], ['HEARD', c.heard]]) {
        if (!hits.length) continue;
        const t = el('h2', title); t.style.margin = '12px 0 0';
        card.append(t);
        for (const hit of hits) {
          const row = el('div', null, 'row');
          if (hit.thumb) {
            const im = el('img'); im.src = '/thumb?f=' + hit.thumb;
            im.style.opacity = 1; row.append(im);
          }
          const col = el('div');
          col.append(el('div', hit.when + ' \\u00b7 ' + (hit.what || hit.label || 'unknown')));
          const tail = (hit.where ? hit.where + ' \\u00b7 ' : '') + 'score ' + hit.score;
          col.append(el('div', tail, 'meta'));
          row.append(col);
          card.append(row);
        }
      }
    }
    box.append(card);
  }

  (async function poll() {
    for (;;) {
      try { render(await (await fetch('/state')).json()); } catch (e) {}
      await new Promise(r => setTimeout(r, 250));
    }
  })();
  // The live crop is the only part that needs its own request. Once a second
  // is plenty for a side-by-side comparison; the video is the live view.
  setInterval(() => { $('thumbB').src = '/crop.jpg?' + Date.now(); }, 1000);
  $('thumbB').onload  = () => { $('thumbB').style.opacity = 1; };
  $('thumbB').onerror = () => { $('thumbB').style.opacity = 0; };

  // Hold TEACH / ASK to record from THIS device's mic and upload one WAV.
  const WORKLET = URL.createObjectURL(new Blob([
    "class Rec extends AudioWorkletProcessor{process(i){const c=i[0][0];" +
    "if(c)this.port.postMessage(c.slice(0));return true}}" +
    "registerProcessor('rec',Rec)"], {type:'text/javascript'}));
  let ctx, stream, chunks = [], holding = false, rate = 16000;

  // Open the mic ONCE and keep it. This used to run per press: getUserMedia,
  // a fresh AudioContext and a worklet compile, all after the finger was
  // already down, and nothing is captured until they finish. Speak promptly
  // and the front of the word is simply not in the file. Keeping it warm also
  // spares the page that work on every press, which is main-thread work in the
  // same tab that is rendering the MJPEG feed.
  // Asking for 16 kHz is what Whisper wants anyway, and it means the browser
  // resamples rather than sending 3x the bytes up the same Wi-Fi link the feed
  // is coming down. Browsers that ignore the option report their real rate in
  // ctx.sampleRate, so the WAV header stays honest either way.
  let micReady;
  function mic() {
    // Memoize the promise, not the context: ctx is only assigned after three
    // awaits, so an `if (ctx) return` guard lets the warm-up listener and the
    // first button press both get past it, and two worklet nodes pushing into
    // one chunk list is audio recorded twice over. On failure, forget it so the
    // next press can retry.
    if (!micReady) micReady = openMic().catch(e => { micReady = null; throw e; });
    return micReady;
  }
  async function openMic() {
    // Replacing a dead one: close it first, or a demo that loses the mic a few
    // times walks into the browser's limit on live AudioContexts.
    if (ctx) { ctx.close().catch(() => {}); ctx = null; }
    stream = await navigator.mediaDevices.getUserMedia(
      { audio: { echoCancellation: true, noiseSuppression: true } });
    ctx = new (window.AudioContext || window.webkitAudioContext)(
      { sampleRate: 16000 });
    await ctx.resume();
    await ctx.audioWorklet.addModule(WORKLET);
    rate = ctx.sampleRate;
    const node = new AudioWorkletNode(ctx, 'rec');
    node.port.onmessage = e => { if (holding) chunks.push(e.data); };
    ctx.createMediaStreamSource(stream).connect(node);
    // A kept mic can still die under you: the track ends if permission is
    // pulled or the input device changes. Forget the memo so the next press
    // opens a fresh one, rather than recording silence behind a healthy-looking
    // indicator until someone reloads the tab. The per-press code that this
    // replaced got that for free.
    stream.getTracks().forEach(t => { t.onended = () => { micReady = null; }; });
  }
  // Warm it on the first touch anywhere, so the first TEACH of the demo is not
  // the one that misses its own first word. Both getUserMedia and resume() want
  // a user gesture, so this cannot happen at load.
  addEventListener('pointerdown', () => mic().catch(() => {}), { once: true });

  async function start(kind, btn) {
    if (holding) return;
    holding = true; chunks = []; btn.classList.add('rec');
    fetch('/listen?k=' + kind);   // narrate "LISTENING" on the filmed view
    try {
      await mic();
      // A backgrounded tab suspends the context (iOS does it for an incoming
      // call too), and a suspended context records nothing. We are inside a
      // user gesture here, which is the one place resume() is allowed.
      if (ctx.state !== 'running') await ctx.resume();
    } catch (e) {
      holding = false; btn.classList.remove('rec'); alert('mic unavailable: ' + e);
    }
  }
  function stop(kind, btn) {
    if (!holding) return;
    holding = false; btn.classList.remove('rec');
    // The mic stream and the AudioContext stay open on purpose: closing them
    // is what made the next press start late. The browser shows a recording
    // indicator for as long as the tab is open; that is the price.
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


def _answer_line(q, res):
    """The spoken sentence — every value in it is a live payload field.

    Built separately from `_speak` because the panel shows it too: the Jetson
    has no audio hardware, so on the appliance this line is the whole answer.
    """
    seen = res["seen"]
    if not seen:
        return "I didn't see anything like that today."
    if "what did you see" in q.lower():
        # the day-inventory question lists the objects; anything else answers
        # with the top hit. ponytail: one phrase check, not intent parsing —
        # the demo script asks this exact question
        labels = [h.payload.get("label") or "something" for h in seen[:3]]
        names = (", ".join(labels[:-1]) + f" and {labels[-1]}"
                 if len(labels) > 1 else labels[0])
        return f"Today I saw {names}."
    p = seen[0].payload
    when = time.strftime("%-I:%M %p", time.localtime(p["ts"]))
    line = f"I saw {p.get('label') or 'something'} at {when}"
    if p.get("where"):
        line += f", in {p['where']}"
    return line + "."


def _hit_json(hit):
    """One recall result, flattened for the panel."""
    p = hit.payload
    return {
        "when": time.strftime("%H:%M", time.localtime(p["ts"])),
        "label": p.get("label"),
        "what": p.get("transcript"),
        "where": p.get("where"),
        "score": round(hit.score, 2),
        "thumb": Path(p["thumb"]).name if p.get("thumb") else None,
    }


def _speak(q, res):
    """Say the answer out loud on a machine that can. macOS `say`; the Jetson
    has no speaker, and the sentence is on the panel either way."""
    line = _answer_line(q, res)
    if shutil.which("say"):
        subprocess.Popen(["say", line])
    return line


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
            elif self.path == "/state":
                # Everything the panel shows, four times a second. Small enough
                # that it is noise beside the video, and it means the panel is
                # laid out by the browser instead of being drawn at a fixed
                # pixel width into a frame the phone then shrinks.
                body = json.dumps(self.app.state()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/thumb?f="):
                # A taught or remembered view, by filename. `.name` drops any
                # directory part, so nothing outside the thumbs directory is
                # reachable however the query is written.
                name = Path(unquote(self.path.split("=", 1)[1])).name
                path = self.app.robot.thumbs / name
                if path.is_file():
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    # thumb filenames are stamped with time_ns and never
                    # rewritten, so the browser can keep them
                    self.send_header("Cache-Control", "max-age=3600")
                    self.end_headers()
                    self.wfile.write(path.read_bytes())
                else:
                    self.send_response(404)
                    self.end_headers()
            elif self.path.startswith("/crop.jpg"):
                jpeg = self.app.crop_jpeg()
                self.send_response(200 if jpeg else 404)
                if jpeg:
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                if jpeg:
                    self.wfile.write(jpeg)
            elif self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                # Send each composed frame once, and always the newest one.
                # This used to push on a fixed 0.04 s tick while the pump
                # renders at the camera's 10 fps, so 60% of the bytes on the
                # wire were the same frame sent again — free on Ethernet,
                # fatal on the appliance's own hotspot, where the radio is the
                # bottleneck and TCP answers oversubscription by queueing
                # rather than dropping: the feed goes slideshow and the delay
                # grows without bound. Taking the latest shot means a client
                # that falls behind loses frames instead of accumulating lag.
                last = -1
                while not self.app.stop.is_set():
                    shot = self.app.shot
                    if shot is None or shot[0] == last:
                        time.sleep(0.01)
                        continue
                    last, jpeg = shot
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        + jpeg + b"\r\n")
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
        self._banner = None   # the status line; assign via `banner` below
        self._banner_seq = 0
        self.card = None   # ("taught", {...}) or ("answer", (q, results))
        self.mem_count = 0
        self.shot = None   # (seq, jpeg): latest composed view for the stream
        self._seq = 0      # bumped per composed frame; see _publish
        self.keys = queue.Queue()
        self.busy = False  # a voice action is running; ignore T/A meanwhile
        # What the robot attends to lives on the robot (see Robot.attention),
        # decided by the detect thread that owns track state.
        self.pending_crop = None   # crop stashed when a teach starts
        self.pending_track = None  # ...and the track it came from
        self.stop = threading.Event()
        self.frame_at = None       # monotonic stamp of the last frame (_watchdog)

    # Every message the panel shows is assigned to `self.banner` from a dozen
    # places. It is a property only so each assignment can bump a counter: the
    # page hides a status line a few seconds after it arrives, and without one
    # it cannot tell a stale message from the same message sent again. Press
    # IGNORE twice a minute apart and the second press would be silent.
    @property
    def banner(self):
        return self._banner

    @banner.setter
    def banner(self, msg):
        self._banner = msg
        self._banner_seq += 1

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
        """The stream carries the annotated camera view and nothing else.

        The panel used to be drawn here and hstacked on, which pinned it to
        480 px however small the screen was. It is HTML now (see PAGE and
        `state`).

        The bandwidth saving is real but small, and it is not the reason: A/B
        on one scene, 49 frames in 5 s either way, 241 KB per composed frame
        against 210 KB per feed frame — about 13%. A flat panel of text is
        cheap to encode; the camera view was always the expensive part.
        """
        view = draw_feed(frame.copy(), tracks, focused)
        ok, buf = cv2.imencode(".jpg", view,
                               [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
        if ok:
            self._publish(buf)

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
                "score": round(focus.score, 3),
                "note": focus.note,
                "thumb": Path(focus.thumb).name if focus.thumb else None,
            } if seen else None,
            "card": self._card_json(),
            "events": r.events[-3:],
        }

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
        return {"kind": "answer", "q": q, "say": _answer_line(q, res),
                "seen": [_hit_json(h) for h in res["seen"][:3]],
                "heard": [_hit_json(h) for h in res["heard"][:3]]}

    def crop_jpeg(self):
        """The live crop of whatever holds focus — the "sees now" half of the
        match. None when nothing is in focus, which the page reads as 404."""
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

    def _publish(self, buf):
        """Hand a composed JPEG to the stream handler, stamped with a sequence
        number so it can send each frame exactly once.

        One tuple, assigned in one bytecode: the number and the bytes can never
        disagree, which a pair of separate attributes could. Only the pump
        thread publishes (splash runs before it starts), so the bump needs no
        lock.
        """
        self._seq += 1
        self.shot = (self._seq, buf.tobytes())

    def _voice_action(self, kind, crop):
        """Laptop escape hatch: record via sounddevice, then process. The
        phone path uploads its own WAV and calls _process directly."""
        wav = UTTERANCE_WAV
        self.banner = "LISTENING · speak now"
        # NO early encoder warm here, unlike on_listen: a cold model load
        # holds the GIL in 1.4-1.6 s blocks (measured), and record_wav below
        # must re-enter stream.read every 100 ms — warming during the
        # recording overflows PortAudio's buffer and chops the audio. The
        # phone path records on the phone, so it has nothing to starve.
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
            # Hand Whisper the utterance, not the hold: silence on either end
            # both slows it down and makes it hallucinate. See trim_to_speech.
            kept = audio.trim_to_speech(wav)
            if kept:
                print(f"trimmed the hold down to {kept:.1f} s of speech")
            # The models this action needs load lazily, and the warm that
            # started at pointerdown may still be running — this call waits on
            # its lock rather than building the same model twice. Before the
            # transcribe, so Whisper is certainly loaded; before the live-state
            # lock, so a first-of-session load can't freeze the detect thread
            # long enough to start killing tracks.
            models.warm_encoders(kind)
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

    def _warm_quietly(self, kind):
        """warm_encoders for a background thread: a failed load (transient
        download, full disk) must be one log line, not a traceback storm —
        the action itself retries the load and reports its own failure."""
        try:
            models.warm_encoders(kind)
        except Exception as e:
            print(f"encoder warm failed (the action will retry): {e}")

    def on_listen(self, kind):
        """Phone started hold-to-talk. Narrate LISTENING and, for teach,
        stash the crop in focus now — the object may drift before release."""
        self.banner = "LISTENING · speak now"
        # Start loading Whisper (and Nomic) NOW, under the hold: the first
        # voice action of a session otherwise pays ~7 s of model loads after
        # the finger lifts. _process re-calls this and waits on its lock, so
        # a quick release cannot race the warm into a double build. Honest
        # cost, once per session: a cold load holds the GIL in 1.4-1.6 s
        # blocks, so the feed pauses while "LISTENING" is up instead of after
        # release behind "thinking...". Relocated, not removed. Safe for
        # tracking because the hold is ~3 s against DEAD_SECONDS = 5.0 —
        # tracks ride it out; that margin is the load-bearing part.
        threading.Thread(target=self._warm_quietly, args=(kind,),
                         daemon=True).start()
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
            img = np.full((720, 1280, 3), BG, np.uint8)
            _text(img, msg, (380, 350), 1.1, INK, 2)
            _text(img, "first run downloads the models (~1.5 GB)",
                  (380, 400), 0.7, VIOLET, 1)
            ok, buf = cv2.imencode(".jpg", img)
            if ok:
                self._publish(buf)

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
