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
             M / MEMORY      everything it knows, with pictures: delete an
                             object from there, or un-ignore one
             Q / IGNORE      dismiss the current unknown (clutter you won't teach)
             F               delete what it knows about the object in focus
                             (keyboard only: a delete aimed by camera is what
                             the MEMORY tab replaces — see its section in the
                             README)
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
from urllib.parse import parse_qs, unquote, urlparse

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
ORANGE = (30, 140, 235)  # the "maybe" band: close to the bar, not over it
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
          --violet:#6047ff; --red:#dc244c; --dim:#6b7280; --line:#d9e0e4;
          --orange:#d97706 }
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
  /* One recall sighting: the object's name above a memory-tab-style picture
     (box + margin, contain not cover -- same reasoning as the tab's cards:
     these are tight crops of arbitrary shape, and cover would cut the ends
     off the object the picture exists to show). */
  .sight { margin-top:12px; font-size:14px }
  .sight .lbl { font-weight:700 }
  .sight .shot { display:flex; gap:10px; align-items:flex-start;
                 margin-top:4px }
  .sight .shot img { width:150px; aspect-ratio:4/3; object-fit:contain;
                     border-radius:8px; background:var(--line);
                     flex:0 0 auto }
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
  #memory,#ignore { min-height:44px; font-size:14px; background:var(--dim) }
  #memory { background:var(--ink) }
  #teach.off { opacity:.4 }
  button.rec { box-shadow:0 0 0 4px #fff inset; filter:brightness(1.25) }

  /* The memory tab: a full-screen sheet, not a third column. A phone has no
     width to spare, and browsing what the robot knows is a thing you stop to
     do rather than something to watch alongside the feed. */
  #mem { position:fixed; inset:0; z-index:9; background:var(--panel);
         display:none; flex-direction:column }
  #mem.on { display:flex }
  #memhead { flex:0 0 auto; display:flex; align-items:center; gap:10px;
             border-bottom:2px solid var(--line); background:#e2e8ea;
             padding:calc(10px + env(safe-area-inset-top)) 12px 10px }
  #memhead h1 { flex:0 0 auto }
  #memmeta { flex:1; font-size:13px; color:var(--dim) }
  #memclose { background:var(--ink); min-height:40px; padding:0 18px;
              font-size:14px }
  /* auto-fill rather than a breakpoint: two columns on a phone, more on a
     laptop, and no width where a card is absurdly wide or unreadably narrow.
     One full-width card per row filled a phone screen with 2.4 objects, which
     is not a list you can scan. */
  /* grid-auto-rows:max-content is load-bearing, not tidiness. This grid has a
     definite height (flex:1 inside a column), and with auto rows the browser
     fits the rows INTO that height rather than letting them overflow it: rows
     computed to 132 px against 266 px of card, scrollHeight stayed equal to
     clientHeight, and every card had its buttons clipped off the bottom with
     nothing to scroll to. align-content:start does not prevent it -- there is
     no free space to distribute, the space is negative. Measured, not guessed;
     it looked fine at four objects purely because the row height the browser
     picked happened to exceed the content. */
  #memlist { flex:1; min-height:0; overflow-y:auto;
             -webkit-overflow-scrolling:touch; display:grid; gap:12px;
             align-content:start; grid-auto-rows:max-content; padding:12px;
             max-width:1200px; width:100%; margin:0 auto;
             grid-template-columns:repeat(auto-fill, minmax(170px, 1fr));
             padding-bottom:calc(24px + env(safe-area-inset-bottom)) }
  .obj { border:2px solid var(--line); border-radius:12px; background:#fff;
         overflow:hidden }
  /* The object with a margin around it, rather than the gray-masked fragment
     CLIP compares. `contain`, not `cover`: these are tight crops of arbitrary
     shape (a standing person is twice as tall as it is wide), and `cover`
     would fill the box by cutting the ends off the very object the picture
     exists to show. 4/3 rather than 16/9 because it wastes less width on the
     tall ones. */
  .obj img { display:block; width:100%; aspect-ratio:4/3; object-fit:contain;
             background:var(--line) }
  .obj .body { padding:9px 11px }
  .name { font-size:17px; font-weight:700; word-break:break-word }
  .said { font-size:13px; color:var(--dim); font-style:italic; margin-top:3px }
  /* Three rows, so cards in a grid line up instead of each deciding for
     itself whether "7 sightings" left room beside a button: the count gets a
     full-width line of its own, RENAME and FORGET share the next, and DROP
     (only on multi-view objects) takes the last. */
  .foot { display:flex; align-items:center; gap:8px; margin-top:9px;
          flex-wrap:wrap }
  .foot .meta { flex:1 0 100% }
  /* Outlined until armed, then filled red: a grid of solid buttons reads as a
     page about deleting things, and every one of them takes two taps anyway.
     Armed text is short ("SURE?") because a 71 px button on a phone cannot
     hold more without clipping. */
  .obj button { flex:1 1 0; min-width:0; min-height:36px; padding:0 8px;
                font-size:12px; background:#fff; color:var(--violet);
                border:2px solid var(--violet) }
  /* every modifier below has to out-specify `.obj button` above, hence the
     repeated prefix: a bare `.del` loses to it and renders violet, and a bare
     `.drop` loses its own row and squeezes onto the line with the others */
  .obj button.del { color:var(--red); border-color:var(--red) }
  .obj button.drop { flex:1 0 100%; color:var(--dim);
                     border-color:var(--line) }
  .obj button.un { flex:1 0 100%; color:var(--teal); border-color:var(--teal) }
  .obj button.armed { background:var(--red); color:#fff;
                      border-color:var(--red) }
  .obj button[disabled] { opacity:.5 }
  .edit { width:100%; font-family:inherit; font-size:17px; font-weight:700;
          padding:3px 5px; border-radius:6px; border:2px solid var(--violet) }
  /* The location row: where the robot is, editable, first row of the grid.
     Every button style in this sheet is scoped (.obj button, #memclose), so
     this one needs its own or it renders as bare white-on-white text. */
  #loc { grid-column:1/-1; display:flex; align-items:center; gap:10px;
         min-height:40px; font-size:14px; color:var(--ink) }
  #loc button { flex:0 0 auto; min-height:36px; padding:0 12px;
                font-size:12px; background:#fff; color:var(--violet);
                border:2px solid var(--violet); border-radius:12px }
  /* 16px is load-bearing, not taste: iOS Safari zooms the page in on any
     focused input whose font is smaller, and does not zoom back out when
     the field goes away. The rename field never triggered it because .edit
     is 17px; this one at 14px did. */
  #loc .edit { flex:1; font-size:16px; font-weight:400 }
  /* display:contents so the ignored cards join the same grid as the objects
     above them instead of stacking inside one cell. */
  #igns { display:contents }
  #igns h2, .wide { grid-column:1/-1; font-size:13px; letter-spacing:.1em;
                    color:var(--dim); margin:8px 0 0 }
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
      <button id="memory">MEMORY</button>
      <button id="ignore">IGNORE</button>
    </div>
  </div>
</div>
<div id="mem">
  <div id="memhead">
    <h1>MEMORY</h1><span id="memmeta"></span>
    <button id="memclose">CLOSE</button>
  </div>
  <div id="memlist"></div>
</div>
<script>
  const $ = id => document.getElementById(id);
  const img = $('view');
  // paused is set while the memory tab is up, where the feed is neither
  // visible nor wanted: dropping it frees the radio for the thumbnails, and
  // the reconnect below must not fight that by pulling it straight back.
  let paused = false;
  img.onerror = () => setTimeout(() => {
    if (!paused) img.src = '/stream?' + Date.now();
  }, 1000);

  let lock = null;
  const wake = async () => { try { lock = await navigator.wakeLock.request('screen'); } catch (e) {} };
  wake();
  addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') wake(); });

  // Laptop keyboard drives the sounddevice mic on the machine itself. R (close
  // the shard and reload it from disk) is keyboard-only on purpose: it is a
  // presenter's beat, not a control anyone should meet on a phone.
  addEventListener('keydown', e => {
    // Never while typing. The rename field bubbles every keystroke up here, so
    // without this guard a name is a string of robot commands: "water bottle"
    // fires a, t, t, r -- two voice actions and a REBOOT of the shard -- f
    // deletes whatever the camera is looking at, and m closes the sheet, which
    // blurs the field and commits the half-typed name.
    const tag = e.target && e.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || e.target.isContentEditable)
      return;
    const k = e.key.toLowerCase();
    if ('tarfq'.includes(k)) fetch('/key?k=' + k);
    if (k === 'm') $('mem').classList.contains('on') ? closeMem() : openMem();
    if (k === 'escape') closeMem();
  });
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

  let shownSeq = null, statusAt = 0, shownCard = '', curGuess = null;
  function render(s) {
    $('count').textContent = s.count + ' memories';
    $('where').textContent = s.where ? 'here: ' + s.where : '';
    const f = s.focus;
    const known = f && f.label;
    // three states: recognized (teal), a near-miss guess (orange, "hat?" --
    // close to the bar but under it, still teachable), or unknown (red).
    // A guess is tappable: confirming teaches this crop as that name, no
    // voice needed -- the crop and the name are both already on screen.
    curGuess = f && !f.label ? f.guess : null;
    $('label').textContent = f ? (f.label ||
        (curGuess ? curGuess + '? \\u2014 tap to confirm'
                  : 'UNKNOWN \\u2014 hold TEACH'))
                               : 'looking\\u2026';
    $('label').style.cursor = curGuess ? 'pointer' : '';
    $('label').style.background =
      f ? (known ? 'var(--teal)' : f.guess ? 'var(--orange)' : 'var(--red)')
        : 'var(--dim)';
    $('fill').style.width = (f ? pct(f.score) : 0) + '%';
    $('fill').style.background =
      f && f.score >= s.threshold ? 'var(--teal)'
        : f && f.guess ? 'var(--orange)' : 'var(--red)';
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
      // the object's last sightings, newest first (or, for "what did you
      // see today?", one row per object): the name above a memory-tab-style
      // picture, with when and where beside it. The SEEN/HEARD pair this
      // replaces is described in CLAUDE.md, "Recall answered from the
      // wrong space".
      for (const hit of c.hits) {
        const s = el('div', null, 'sight');
        s.append(el('div', hit.label || 'unknown', 'lbl'));
        const shot = el('div', null, 'shot');
        if (hit.thumb) {
          const im = el('img'); im.src = '/thumb?f=' + hit.thumb;
          im.style.opacity = 1; shot.append(im);
        }
        const col = el('div');
        col.append(el('div', hit.when));
        if (hit.where) col.append(el('div', hit.where, 'meta'));
        shot.append(col);
        s.append(shot);
        card.append(s);
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
  // Confirming a guess sends the label the panel DISPLAYED, and the server
  // re-checks it against the live guess before teaching -- the FORGET
  // lesson: focus can move between deciding to tap and tapping.
  $('label').onclick = () => {
    if (curGuess) fetch('/confirm?label=' + encodeURIComponent(curGuess),
                        { method: 'POST' }).catch(() => {});
  };
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

  // ---- the memory tab -----------------------------------------------------
  // What the robot knows, with pictures, and the two things you can do about
  // it: forget an object, or take one off the ignore list. FORGET used to act
  // on whatever the box was on at the instant of the press, which deletes the
  // wrong object whenever focus moves under a finger already on its way down.
  // Here you delete the card you are looking at.
  const PAGE_N = 12;
  const igns = el('div');
  igns.id = 'igns';
  const loc = el('div');
  loc.id = 'loc';
  let locWhere = null;
  let memAt = 0, memMore = true, memBusy = false, memGen = 0;

  function openMem() {
    $('mem').classList.add('on');
    // Let go of the video: on the robot's own hotspot the feed is most of the
    // radio, and every byte of it is now behind an opaque sheet.
    paused = true; img.removeAttribute('src');
    $('memlist').replaceChildren(loc, igns);
    drawLoc();
    // A page fetched before this reload must not land in the list after it.
    // openMem is how every delete and rename redraws, so there is very often
    // one in flight: without the generation counter the stale response inserts
    // the card that was just deleted and leaves memAt at its old offset, so
    // page 0 is never loaded until the tab is closed and reopened. Clearing
    // memBusy here (rather than waiting for that response) is what lets the
    // fresh page start immediately; the stale one checks the counter before it
    // touches anything.
    memAt = 0; memMore = true; memBusy = false; memGen++;
    loadPage();
    loadIgnored();
  }
  function closeMem() {
    if (!$('mem').classList.contains('on')) return;
    $('mem').classList.remove('on');
    paused = false; img.src = '/stream?' + Date.now();
  }
  $('memory').onclick = openMem;
  $('memclose').onclick = closeMem;

  async function loadPage() {
    if (memBusy || !memMore) return;
    memBusy = true;
    const gen = memGen;
    try {
      const r = await (await fetch(
        '/memories?offset=' + memAt + '&limit=' + PAGE_N)).json();
      if (gen !== memGen) return;   // the list was reloaded under us
      $('memmeta').textContent =
        r.total + ' object' + (r.total === 1 ? '' : 's') + ' \\u00b7 ' +
        r.points + ' points';
      locWhere = r.where || null;
      // never redraw over an edit in progress: a scroll can land a page
      // while the field is focused, and replacing it throws the typing away
      if (!loc.contains(document.activeElement)) drawLoc();
      const list = $('memlist');
      // insertBefore, not append: the ignored section lives at the end of the
      // same grid and must stay there as pages arrive above it.
      for (const o of r.objects) list.insertBefore(objCard(o), igns);
      memAt += r.objects.length;
      // Stop on a short page as well as on the count, or a delete racing the
      // scroll leaves this asking for a page that no longer exists forever.
      memMore = r.objects.length === PAGE_N && memAt < r.total;
      if (!r.total) list.insertBefore(
        el('div', 'Nothing taught yet. Point the camera at something and hold '
                  + 'TEACH.', 'wide'), igns);
    // only the current generation owns the flag; a stale page clearing it
    // would let a second fetch start on top of the fresh one
    } catch (e) {} finally { if (gen === memGen) memBusy = false; }
  }
  $('memlist').addEventListener('scroll', () => {
    const n = $('memlist');
    if (n.scrollTop + n.clientHeight > n.scrollHeight - 500) loadPage();
  });

  function objCard(o) {
    const card = el('div', null, 'obj');
    const pic = el('img'), said = el('div', null, 'said');
    const cap = el('div', null, 'meta');
    // One tap-cycle over everything the robot holds for this object: the
    // taught views first, then its own recent photos (sightings, newest
    // first, the same pictures recall answers with -- they used to be
    // reachable only by asking "where did I leave it").
    const sights = o.sightings || [];
    const entries = o.views.concat(sights);
    let i = 0;
    const drop = armedBtn('DROP THIS VIEW', 'drop', () =>
      post('/forget_view?id=' + entries[i].id +
           '&label=' + encodeURIComponent(o.label)));
    const show = () => {
      const v = entries[i], taught = i < o.views.length;
      pic.src = '/thumb?f=' + encodeURIComponent(v.scene || v.thumb || '');
      said.textContent = v.transcript ? '\\u201c' + v.transcript + '\\u201d' : '';
      // one running count over the whole cycle; the word says which kind
      cap.textContent = (i + 1) + ' of ' + entries.length +
        ' \\u00b7 ' + (taught ? 'taught' : 'seen') +
        ' \\u00b7 ' + v.when + (v.where ? ' \\u00b7 ' + v.where : '');
      // Sightings are droppable too (a stale-box photo is worth removing);
      // only a taught view hides the button when it is the object's last
      // one, because dropping that forgets the whole object and FORGET is
      // the honest button for it.
      drop.style.display = (taught && o.views.length < 2) ? 'none' : '';
    };
    // Tap the picture to walk the cycle. Re-teaching adds a view rather than
    // replacing it -- that is what makes recognition work from more than one
    // angle -- and this is the only way to see whether the second one was any
    // good. DROP THIS VIEW acts on whichever taught view is showing.
    // disarm on the way: an armed button plus a tap on the picture would
    // delete a different view than the one that was armed
    pic.onclick = () => {
      i = (i + 1) % entries.length; show(); drop.disarm();
    };
    show();
    const name = el('div', o.label, 'name');
    const foot = el('div', null, 'foot');
    foot.append(el('div', o.seen + ' sighting' + (o.seen === 1 ? '' : 's'), 'meta'),
                renameBtn(o.label, name),
                armedBtn('FORGET', 'del', () =>
                  post('/forget?label=' + encodeURIComponent(o.label))));
    // always in the row now that sightings are droppable; show() hides it on
    // the one case where dropping means forgetting (a single taught view)
    foot.append(drop);
    const body = el('div', null, 'body');
    body.append(name, said, cap, foot);
    card.append(pic, body);
    return card;
  }

  const post = url => fetch(url, { method: 'POST' }).catch(() => {});

  // ---- location: where the robot is -----------------------------------
  // Stamped on every memory written from here on, and read back by recall
  // ("I saw my keys at 2:14 PM, in the hotel room"). --location on the
  // command line was the only way to set it, which on the appliance meant
  // editing the service file; this row is the live version. The value is
  // persisted beside the shard, so it survives the power cut that is the
  // appliance's off switch. Old memories keep the place they were taught at.
  function drawLoc() {
    const b = el('button', locWhere ? 'CHANGE' : 'SET LOCATION');
    b.onclick = editLoc;
    loc.replaceChildren(
      el('span', locWhere ? 'here: ' + locWhere : 'location not set'), b);
  }
  function editLoc() {
    const inp = document.createElement('input');
    // same shape as the rename field, traps included: the global keydown
    // listener already bails on INPUT targets, so typing here cannot fire
    // robot commands
    inp.className = 'edit'; inp.value = locWhere || ''; inp.maxLength = 40;
    inp.placeholder = 'e.g. hotel room, booth 12';
    inp.autocapitalize = 'none'; inp.spellcheck = false;
    loc.replaceChildren(inp);
    inp.focus(); inp.select();
    let done = false;
    // one guard for both exits, exactly as in renameBtn: Enter fires, then
    // the blur it causes fires again
    const finish = async save => {
      if (done) return;
      done = true;
      const to = inp.value.trim();
      if (save && to !== (locWhere || '')) {
        // an empty value is a real request: it clears the location
        await post('/where?to=' + encodeURIComponent(to));
        openMem();  // redraw from the robot's answer, not our guess
      } else {
        drawLoc();
      }
    };
    inp.onkeydown = e => {
      if (e.key === 'Enter') finish(true);
      if (e.key === 'Escape') finish(false);
    };
    inp.onblur = () => finish(true);
  }

  function armedBtn(text, cls, action) {
    const b = el('button', text, cls);
    let armed = 0;
    b.disarm = () => {
      armed = 0; b.textContent = text; b.classList.remove('armed');
    };
    b.onclick = async () => {
      // Arm on the first tap, act on the second. A list you scroll with your
      // thumb is a list you will tap by accident, and neither of these can be
      // undone -- the vectors are gone.
      if (Date.now() > armed) {
        armed = Date.now() + 4000;
        b.textContent = 'SURE?';
        b.classList.add('armed');
        setTimeout(() => {
          if (Date.now() > armed) {
            b.textContent = text; b.classList.remove('armed');
          }
        }, 4200);
        return;
      }
      b.disabled = true;
      await action();
      openMem();  // redraw from the shard, so the count is the robot's, not ours
    };
    return b;
  }

  // Rename in place: the name becomes a text field. This is the cure for
  // whisper-base hearing "laptop" as "La-caw" -- the vectors were fine, only
  // the word was wrong, and forgetting the object to fix a typo threw away
  // every view of it.
  function renameBtn(label, name) {
    const b = el('button', 'RENAME', 'ren');
    b.onclick = () => {
      const inp = document.createElement('input');
      inp.className = 'edit'; inp.value = label; inp.maxLength = 40;
      inp.autocapitalize = 'none'; inp.spellcheck = false;
      name.replaceWith(inp);
      inp.focus(); inp.select();
      let done = false;
      // one guard for both exits: Enter fires, then the blur it causes fires
      // again, and the second one would post the rename twice
      const finish = async save => {
        if (done) return;
        done = true;
        const to = inp.value.trim();
        // compared exactly, not case-folded: fixing the capitals is a real
        // rename on a shard written before labels were lowercased, and it is
        // the only way to change what an old card displays
        if (save && to && to !== label) {
          await post('/rename?label=' + encodeURIComponent(label) +
                     '&to=' + encodeURIComponent(to));
          openMem();
        } else {
          inp.replaceWith(name);
        }
      };
      inp.onkeydown = e => {
        if (e.key === 'Enter') finish(true);
        if (e.key === 'Escape') finish(false);
      };
      inp.onblur = () => finish(true);
    };
    return b;
  }

  async function loadIgnored() {
    igns.replaceChildren();
    const gen = memGen;
    let r;
    try { r = await (await fetch('/ignored')).json(); } catch (e) { return; }
    if (gen !== memGen) return;   // same staleness guard as loadPage
    if (!r.ignored.length) return;
    igns.append(el('h2', 'IGNORED \\u00b7 ' + r.ignored.length));
    for (const g of r.ignored) {
      const card = el('div', null, 'obj');
      const pic = el('img');
      if (g.thumb) pic.src = '/thumb?f=' + encodeURIComponent(g.thumb);
      const b = el('button', 'TRACK AGAIN', 'un');
      b.onclick = async () => {
        b.disabled = true;
        // pid stays the string the server sent -- point ids exceed 2^53
        try { await fetch('/unignore?pid=' + g.pid, { method: 'POST' }); } catch (e) {}
        loadIgnored();
      };
      const foot = el('div', null, 'foot');
      foot.append(el('div', 'ignored ' + g.when, 'meta'), b);
      const body = el('div', null, 'body');
      body.append(foot);
      card.append(pic, body);
      igns.append(card);
    }
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
        # three states, not two: a near-miss (within MAYBE_MARGIN of the
        # bar) is orange and says which taught object it almost was —
        # "hat? 0.87" — instead of a bare UNKNOWN. Display only; the track
        # is still unlabeled, still the teach target.
        color = TEAL if known else (ORANGE if t.guess else RED)
        thick = 6 if t is focused else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        if t.last_query:
            tag = (f"{t.label}  {t.score:.2f}" if known
                   else f"{t.guess}?  {t.score:.2f}" if t.guess
                   else f"UNKNOWN  {t.score:.2f}")
            _chip(frame, tag, (x1 + 4, max(30, y1 - 12)), color)
    return frame


def _when(ts):
    """A spoken timestamp: today's sightings keep just the clock, older ones
    name the day — recall is no longer capped at midnight, and "I saw it at
    9:12 PM" is a lie by omission when that was Tuesday."""
    t = time.localtime(ts)
    if time.strftime("%Y%m%d", t) == time.strftime("%Y%m%d"):
        return time.strftime("%-I:%M %p", t)
    return time.strftime("%b %-d, %-I:%M %p", t)


def _answer_line(q, res):
    """The spoken sentence — every value in it is a live payload field.

    Built separately from `_speak` because the panel shows it too: the Jetson
    has no audio hardware, so on the appliance this line is the whole answer.
    """
    s = res["sightings"]
    if res["inventory"]:
        if not s:
            return "I didn't see anything today."
        labels = [h.payload.get("label") or "something" for h in s[:3]]
        names = (", ".join(labels[:-1]) + f" and {labels[-1]}"
                 if len(labels) > 1 else labels[0])
        return f"Today I saw {names}."
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
    """One sighting row, flattened for the panel. No score: these come from
    a scroll ordered by time, not a vector search — see Robot.ask."""
    p = hit.payload
    return {
        "when": time.strftime("%b %-d, %H:%M", time.localtime(p["ts"])),
        "label": p.get("label"),
        "where": p.get("where"),
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

    def _query(self, key, default=None):
        """One query-string value, or `default`."""
        vals = parse_qs(urlparse(self.path).query).get(key)
        return vals[0] if vals else default

    def _int(self, key, default):
        """A non-negative integer query value. Anything else is the default —
        these arrive from a URL, and a typo should not be a traceback. Pass a
        non-negative default: the clamp applies to it too."""
        try:
            return max(0, int(self._query(key, default)))
        except (TypeError, ValueError):
            return default

    def _send_json(self, obj, status=200):
        """Content-Length on every JSON reply, and never cached: a frozen
        panel beside a moving video is a horrible thing to diagnose in a room,
        and a stale memory list would still offer an object just deleted."""
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

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
                self._send_json(self.app.state())
            elif self.path.startswith("/memories"):
                # One page of the memory tab. Paged so the list can be scrolled
                # rather than built in one lump on a phone.
                self._send_json(self.app.memories(
                    offset=self._int("offset", 0),
                    limit=self._int("limit", 12)))
            elif self.path.startswith("/ignored"):
                self._send_json(self.app.ignored())
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
            # The two mutations the memory tab makes. POST, and answered with
            # the result rather than a 204: the tab redraws from what the robot
            # says happened, so a delete that found nothing cannot leave a card
            # on screen that no longer exists.
            # /forget_view first: it also starts with "/forget"
            elif self.path.startswith("/forget_view"):
                label, pid = self._query("label"), self._int("id", 0)
                self._send_json(
                    self.app.forget_view(pid, label)
                    if label and pid else {"n": 0},
                    200 if label and pid else 400)
            elif self.path.startswith("/forget"):
                label = self._query("label")
                self._send_json(
                    {"label": label, "n": self.app.forget_label(label)}
                    if label else {"n": 0}, 200 if label else 400)
            elif self.path.startswith("/rename"):
                label, to = self._query("label"), self._query("to", "")
                # capped here, not only in the page (which stops at 40): a
                # label is a key that recall reads out loud and every card
                # shows, and nothing else limits what a POST can write
                to = " ".join((to or "").split())[:60]
                self._send_json(
                    {"label": label, "to": to,
                     "n": self.app.rename(label, to)}
                    if label and to else {"n": 0}, 200 if label and to else 400)
            elif self.path.startswith("/confirm"):
                # tap on the orange "dylan?" — teach that crop as that name
                label = self._query("label")
                self._send_json(self.app.confirm(label)
                                if label else {"ok": False},
                                200 if label else 400)
            elif self.path.startswith("/unignore"):
                # the pid travels as a string (point ids exceed 2^53, see
                # _view_json); 0 on a malformed request misses harmlessly
                pid = self._int("pid", 0)
                self._send_json({"pid": str(pid),
                                 "ok": self.app.unignore(pid)})
            elif self.path.startswith("/where"):
                # the device moved: change the place stamped on new memories.
                # An empty (or absent) value clears it. Normalized and capped
                # server-side like /rename, and for the same reason — the
                # page's maxLength means nothing to a crafted POST, and recall
                # reads this string out loud.
                self._send_json({"where": self.app.set_where(
                    self._query("to", ""))})
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
        self.pending_teach = None  # (crop, frame, box) stashed when a teach starts
        self.pending_track = None  # ...and the track they came from
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

        The whole point of the tab: FORGET used to act on whatever the box
        happened to be on, so a press that landed as focus moved deleted the
        wrong object — and there was no way to see what had been learned
        without pointing the camera at it again.

        The grouping scans every taught point per page, which is single digits
        of points at demo scale. If a shard ever holds hundreds of objects,
        page the scroll itself rather than slicing here.

        No app lock, deliberately: Memory serializes its own shard access,
        including a REBOOT's reopen, so the scan cannot race it. The app lock
        is held by the detect thread for a whole YOLO pass, and behind it this
        page cost 0.8-1.6 s (measured) for a half-millisecond scan — paid on
        every open, delete and rename, since all of them redraw the tab.
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
                # the robot's own recent photos of it, same shape as a view
                # (a sighting has no transcript; its picture is the thumb).
                # The card's tap-cycle walks these after the taught views —
                # they are the pictures recall answers with, and until they
                # were here the tab had no way to show them.
                "sightings": [self._view_json(v) for v in o["sightings"]],
            } for o in objects[offset:offset + limit]],
        }

    @staticmethod
    def _view_json(payload):
        """One taught view. `scene` is the unmasked picture of the object with
        a margin around it (see Robot._scene) and falls back to the crop, so
        points written before scenes existed still show a picture rather than
        a hole."""
        thumb = payload.get("thumb")
        scene = payload.get("scene") or thumb
        return {
            # A STRING on purpose, so one view can be dropped on its own.
            # Point ids come from itertools.count(time.time_ns()), so they are
            # around 1.8e18 — past JavaScript's 2^53 safe integer, where JSON
            # numbers silently round. The page would then ask to delete an id
            # that does not exist and the view would quietly survive its own
            # deletion. As text it round-trips exactly; the server parses it
            # back to an int.
            "id": str(payload["id"]),
            "when": time.strftime("%b %-d, %H:%M",
                                  time.localtime(payload.get("ts") or 0)),
            "transcript": payload.get("transcript"),
            "where": payload.get("where"),
            "thumb": Path(thumb).name if thumb else None,
            "scene": Path(scene).name if scene else None,
        }

    def ignored(self):
        """Everything IGNORE has dismissed, for the tab to list and undo.

        Read from the shard now (kind="ignored" points), not the detector's
        session dict — dismissals persist across restarts, so the list must
        too. No app lock, for the same reason as `memories`: Memory
        serializes its own shard access, and behind the app lock this read
        queued up to 1.6 s behind a YOLO pass. The pid is a string for the
        same reason every point id this page sees is (see _view_json).
        """
        rows = self.robot.memory.ignored()
        return {"ignored": [{
            "pid": str(r.id),
            "when": time.strftime("%b %-d, %H:%M",
                                  time.localtime(r.payload.get("ts") or 0)),
            "thumb": (Path(r.payload["thumb"]).name
                      if r.payload.get("thumb") else None),
        } for r in rows]}

    def forget_label(self, label):
        """Delete one object by name — from the tab, or from the F key."""
        with self.lock:
            n = self.robot.forget(label)
            self.mem_count = self.robot.memory.count()
        self.card = ("forgot", (label, n))
        self.banner = f'forgot "{label}"'
        return n

    def rename(self, label, to):
        """Rename one object, from the tab. No vectors move — see Robot.rename."""
        with self.lock:
            n = self.robot.rename(label, to)
        self.banner = f'renamed to "{to}"'
        return n

    def forget_view(self, pid, label):
        """Drop one taught view of an object, from the tab."""
        with self.lock:
            n, whole = self.robot.forget_view(pid, label)
            self.mem_count = self.robot.memory.count()
        if whole:
            # it was the last view, so the object itself is gone — say so with
            # the same card FORGET uses, or the tab looks like it over-deleted
            self.card = ("forgot", (label, n))
            self.banner = f'that was the last view of "{label}" — forgot it'
        elif n:
            self.banner = f'dropped one view of "{label}"'
        else:
            # the page was looking at a list that has since changed
            self.banner = "that view is already gone"
        return {"n": n, "whole": whole, "label": label}

    def unignore(self, pid):
        with self.lock:
            ok = self.robot.unignore(pid)
        self.banner = "tracking that again" if ok else "that isn't ignored"
        return ok

    def confirm(self, label):
        """Tap on the orange guess: teach the attending crop as that name.

        Encoders warm OUTSIDE the lock, exactly like the voice path — the
        first action of a session pays ~7 s of model loads, and paying them
        under the lock stalls the detect thread long enough to start killing
        tracks. Warm sessions pay ~0.3 s of embeds.
        """
        self.banner = "thinking..."
        models.warm_encoders()
        with self.lock:
            res = self.robot.confirm(label, frame=self.latest)
            if res:
                self.mem_count = self.robot.memory.count()
        if res is None:
            # the guess moved on between the paint and the tap — refuse
            # honestly rather than teach whatever holds the panel now
            self.banner = "nothing to confirm"
            return {"ok": False}
        self.card = ("taught", res)
        self.banner = f'taught: "{res["label"]}"'
        return {"ok": True, "label": res["label"]}

    def set_where(self, place):
        """Update the robot's location, from the tab. Under the lock like the
        other mutations — it is one write, but a teach mid-flight reads
        `memory.where` and should see one value, not a torn decision."""
        with self.lock:
            where = self.robot.set_where(place)
        self.banner = f'here: "{where}"' if where else "location cleared"
        return where

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
                "hits": [_hit_json(h) for h in res["sightings"][:3]]}

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

    def _teach_target(self, track):
        """Everything a teach needs from the live view, grabbed in one go:
        the crop that gets embedded, and the whole frame it sits in for the
        memory tab to show later. None when there is nothing to teach.

        Copied, not referenced: the detect thread keeps rewriting a track's
        crop, and the object may well have moved by the time the finger lifts
        six seconds later.
        """
        if track is None or track.crop is None:
            return None
        frame = self.latest
        return (track.crop.copy(),
                None if frame is None else frame.copy(),
                track.box)

    def _voice_action(self, kind, target):
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
        self._process(kind, wav, target, heard=spoke)

    def _phone_audio(self, kind, body, target):
        """Phone hold-to-talk: the uploaded WAV replaces the sounddevice
        recording; everything downstream is identical to the laptop mic."""
        wav = UTTERANCE_WAV
        with open(wav, "wb") as f:
            f.write(body)
        self._process(kind, wav, target)

    def _process(self, kind, wav, target, heard=None):
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
            models.warm_encoders()
            # Whisper takes seconds; run it before claiming the lock so the
            # detector keeps tracking while the robot listens.
            q = models.transcribe(wav)
            if kind == "t":
                crop, frame, box = target
                with self.lock:
                    taught = self.robot.teach(crop, q, frame=frame, box=box)
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
                # No app lock: ask reads only memory, which serializes its
                # own shard access, and touches no track state. Holding the
                # app lock here just queued the answer behind a YOLO pass —
                # up to a second of wait on a path the operator is watching.
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

    def _warm_quietly(self):
        """warm_encoders for a background thread: a failed load (transient
        download, full disk) must be one log line, not a traceback storm —
        the action itself retries the load and reports its own failure."""
        try:
            models.warm_encoders()
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
        threading.Thread(target=self._warm_quietly, daemon=True).start()
        if kind == "t":
            f = self.robot.teachable
            self.pending_teach = self._teach_target(f)
            self.pending_track = f if self.pending_teach else None

    def on_audio(self, kind, body):
        """Phone released hold-to-talk with a WAV. Returns an HTTP status.
        Mirrors the busy-gate in `_handle_key`. Serialized by the single phone
        UI in the demo; add a lock if multiple clients are ever allowed — the
        check-and-set on `busy` below is not atomic."""
        if self.busy:
            return 409
        if kind == "t" and self.pending_teach is None:
            self.banner = "nothing in focus to teach"
            return 409
        self.busy = True
        target = self.pending_teach if kind == "t" else None
        threading.Thread(target=self._phone_audio, args=(kind, body, target),
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
            target = self._teach_target(teachable)
            if target is None:
                self.banner = "nothing new to teach"
                return
            self.busy = True
            # remember which track this taught, so only that one re-asks
            # memory afterwards instead of every box on screen
            self.pending_track = teachable
            threading.Thread(target=self._voice_action, args=("t", target),
                             daemon=True).start()
        elif key == "a" and not self.busy:
            self.busy = True
            threading.Thread(target=self._voice_action,
                             args=("a", None),
                             daemon=True).start()
        elif key == "f":
            # forget the recognized object the panel is showing (focused).
            # Keyboard only now: aiming a delete with the camera is what the
            # memory tab exists to replace, but it is still the fastest beat to
            # film, and a key is not something a thumb lands on by accident.
            if focused is None or not focused.label:
                self.banner = "nothing recognized to forget"
            else:
                self.forget_label(focused.label)
        elif key == "q":
            # dismiss the current unknown so the robot stops offering it
            if teachable is None:
                self.banner = "no unknown to ignore"
            else:
                with self.lock:
                    self.robot.ignore(teachable, self.latest)
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
                         'points back to where it learned them. Overrides the '
                         'value set from the memory tab (which persists in '
                         'the data dir) for this run only')
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
