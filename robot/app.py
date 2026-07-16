"""Live demo app: webcam + mic, or headless frame replay.

    uv run python -m robot.app                 # live (default camera + mic)
    uv run python -m robot.app --source DIR    # headless: replay images/video

Keys (live): T teach the focused object (speak while it listens)
             A ask "what did you see today?" by voice
             R the reboot beat: close the shard, reload from disk, re-ask
             Q quit
"""
import argparse
import sys
import threading
import time
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
WIN = "L6 Robot Memory"


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
               threshold=RECOGNIZE_THRESHOLD):
    panel = np.full((h, PANEL_W, 3), BG, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (PANEL_W, 54), VIOLET, -1)
    _text(panel, "ROBOT MEMORY", (20, 37), 1.0, (255, 255, 255), 2)
    _text(panel, f"{count} memories", (PANEL_W - 190, 37),
          0.7, (255, 255, 255), 1)
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
        for i, line in enumerate(_wrap(f'"{taught["transcript"]}"', 40)[:3]):
            _text(panel, line, (26, y + 58 + 24 * i), 0.58)
    elif card and card[0] == "answer":
        q, res = card[1]
        for line in _wrap(f'Q: "{q}"', 40):
            _text(panel, line, (20, y), 0.65, VIOLET, 1); y += 24
        for group, hits in (("SEEN", res["seen"]), ("HEARD", res["heard"])):
            _text(panel, group, (20, y + 6), 0.7, INK, 2); y += 30
            for hit in hits[:3]:
                p = hit.payload
                what = p.get("transcript") or p.get("label") or "?"
                when = time.strftime("%H:%M", time.localtime(p["ts"]))
                line = _wrap(f"{when}  {what}", 42)[0]
                _text(panel, f"{line}  ({hit.score:.2f})", (30, y), 0.58)
                y += 24
            y += 8
    log = events[-3:]
    ly = h - 44 - 22 * len(log)
    if log:
        _text(panel, "memory writes", (20, ly), 0.55, VIOLET, 1)
    for i, e in enumerate(log):
        _text(panel, e[:44], (20, ly + 22 * (i + 1)), 0.55)
    cv2.rectangle(panel, (0, h - 34), (PANEL_W, h), INK, -1)
    _text(panel, "T teach   A ask   R reboot   Q quit", (20, h - 11),
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
                           self.robot.memory.threshold)
        cv2.imshow(WIN, np.hstack([view, panel]))

    def _listen(self, frame, tracks, focused, seconds):
        wav = "/tmp/l6-utterance.wav"
        self.banner = "LISTENING — speak now"
        self._render(frame, tracks, focused)  # show it before recording blocks
        cv2.waitKey(1)
        audio.record_wav(wav, seconds)
        if audio.is_silent(wav):
            self.banner = "didn't hear anything — try again"
            return None
        self.banner = "thinking..."
        self._render(frame, tracks, focused)
        cv2.waitKey(1)
        return wav

    def run(self):
        sys.setswitchinterval(0.002)  # keeps the feed smooth while YOLO runs
        # WINDOW_NORMAL, created once: the default AUTOSIZE window re-derives
        # its geometry on every imshow, which on multi-monitor macOS snaps it
        # back to the original display — sometimes offscreen.
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1280 + PANEL_W, 720)
        print("warming up models...")
        models.warm_up()
        threading.Thread(target=self._detect_loop, daemon=True).start()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.latest = frame
            tracks = list(self.tracks)
            focused = self.robot.focused(tracks)
            self._render(frame, tracks, focused)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("t") and focused is not None:
                crop = focused.crop.copy()
                wav = self._listen(frame, tracks, focused, 5)
                if wav is None:
                    continue
                with self.lock:
                    taught = self.robot.teach(crop, wav)
                    self.mem_count = self.robot.memory.count()
                    for t in self.robot.detector.tracks.values():
                        t.last_query = 0  # requery now: watch it recognize
                self.card = ("taught", taught)
                self.banner = f'taught: "{taught["label"]}"'
            elif key == ord("a"):
                wav = self._listen(frame, tracks, focused, 4)
                if wav is None:
                    continue
                with self.lock:
                    q, res = self.robot.ask_from_wav(wav)
                self.card = ("answer", (q, res))
                self.banner = None
            elif key == ord("r"):
                with self.lock:
                    n = self.robot.reboot()
                    self.mem_count = n
                    if self.card and self.card[0] == "answer":
                        q = self.card[1][0]
                        self.card = ("answer", (q, self.robot.ask(q)))
                self.banner = f"rebooted from disk — {n} memories"
        self.stop.set()
        self.cap.release()
        self.robot.close()
        cv2.destroyAllWindows()


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
    ap.add_argument("--data", default="edge-data", help="shard directory")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=RECOGNIZE_THRESHOLD,
                    help="recognition threshold (calibration knob; if it "
                         "moves for the shoot, L5 moves with it)")
    ap.add_argument("--conf", type=float, default=None,
                    help="detector confidence floor (raise to track less)")
    args = ap.parse_args()
    robot = Robot(data_dir=args.data, threshold=args.threshold,
                  conf=args.conf)
    if args.source:
        replay(robot, args.source)
    else:
        LiveApp(robot, args.camera).run()


if __name__ == "__main__":
    main()
