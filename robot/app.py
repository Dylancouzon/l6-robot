"""Start the robot.

    uv run python -m robot.app                 # live -> http://127.0.0.1:8765
    uv run python -m robot.app --host 0.0.0.0  # live, reachable from a phone
    uv run python -m robot.app --source DIR    # headless: replay images/video

The live view is a web page rather than a desktop window, which is what lets
a phone be the whole interface. Controls, as keys in the browser tab or as
touch buttons:

    T / hold TEACH   teach the focused unknown object (speak while held)
    A / hold ASK     ask a question by voice
    M / MEMORY       everything it knows, with pictures
    Q / IGNORE       dismiss the current unknown
    F                forget the focused recognized object (keyboard only)
    R                close the shard and reload it from disk (keyboard only)

Quit with Ctrl-C. There is no on-screen quit: a stray tap would end the demo.
"""
import argparse
import ipaddress
import shutil
import time
from pathlib import Path

import cv2

from robot.brain.core import Robot
from robot.brain.memory import RECOGNIZE_THRESHOLD

REPO = Path(__file__).resolve().parent.parent


def replay(robot, source):
    """Headless file mode: a directory of images, or a video file.

    Covers the crop and recognition path. It does NOT cover attention: each
    image is its own scene with one object, so focus stickiness never runs.
    """
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

    The value is written as an `IP:` subjectAltName, and a hostname there makes
    openssl reject the whole certificate - which lands in the HTTPS fallback
    and quietly serves plain http, taking the phone mic away over a typo.
    """
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not an IP address. The certificate needs an IP here; "
            "for the robot's own hotspot that is 10.42.0.1.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="image dir or video file (headless)")
    # resolved against the repo root, not the cwd, so launching from another
    # directory finds the same shard instead of creating a fresh one
    ap.add_argument("--data", default=str(REPO / "edge-data"),
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
                         "(unattended runs only - a laptop lid-close looks "
                         "like a stall)")
    # The next three are per-camera. Their permanent home is .env; these flags
    # override it for a single run, which is what you want while calibrating.
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
                         '"Hotel room", read back by recall. Overrides the '
                         'value set from the memory tab, for this run only')
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
        # imported here, not at the top: headless replay drives the brain
        # only, and has no use for a camera, a server or a microphone
        from robot.device.live import LiveApp
        LiveApp(robot, args.camera, watchdog=args.watchdog).run(
            args.host, advertise=args.advertise)


if __name__ == "__main__":
    main()
