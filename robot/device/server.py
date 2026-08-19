"""The web server: routes, the MJPEG stream, and the certificate.

Every route is a thin call into `LiveApp` (see live.py). The page itself is
page.html, served whole; the browser polls /state four times a second for
everything that is not video.

HTTP/1.0 is deliberate - each poll is its own connection. Moving to 1.1 would
need an accurate Content-Length on every response or clients hang.
"""
import json
import socket
import ssl
import subprocess
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

PAGE = (Path(__file__).resolve().parent / "page.html").read_bytes()


def lan_ip():
    """LAN IP via a dummy UDP connect (no packets sent). Loopback on failure."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert(ip):
    """Self-signed certificate, so the phone browser treats the page as a
    secure context - getUserMedia refuses plain http. Returns (cert, key).

    Three details decide how loudly the browser complains:

    * A subjectAltName for the address you actually open. Browsers stopped
      reading the CN field in 2017, so a CN-only certificate names nothing,
      which is a harsher warning that trusting it cannot fix.
    * Validity under 825 days, or Safari refuses it outright.
    * CA:TRUE, so a phone can install it as a root once and stop warning.

    Built with a config file rather than -addext, which the LibreSSL that
    ships as /usr/bin/openssl on macOS does not have. Regenerated only when
    the address changes, so a trusted phone stays trusted across restarts.
    """
    root = Path(__file__).resolve().parents[2] / "cert"
    cert, key, named = root / "cert.pem", root / "key.pem", root / "names.txt"
    # localhost stays valid so the laptop view works off the same certificate
    want = ",".join([f"IP:{ip}", "IP:127.0.0.1", "DNS:localhost",
                     f"DNS:{socket.gethostname().split('.')[0]}.local"])
    if (cert.exists() and key.exists() and named.exists()
            and named.read_text() == want):
        return str(cert), str(key)
    root.mkdir(exist_ok=True)
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


class StreamHandler(BaseHTTPRequestHandler):
    app = None   # set by LiveApp
    cert = None  # path to the served certificate, when running HTTPS

    def log_message(self, *args):
        pass

    def _query(self, key, default=None):
        """One query-string value, or `default`."""
        vals = parse_qs(urlparse(self.path).query).get(key)
        return vals[0] if vals else default

    def _int(self, key, default):
        """A non-negative integer query value; anything else is the default.
        These arrive from a URL, and a typo should not be a traceback."""
        try:
            return max(0, int(self._query(key, default)))
        except (TypeError, ValueError):
            return default

    def _send_json(self, obj, status=200):
        """Content-Length on every JSON reply, and never cached: a frozen panel
        beside a moving video is horrible to diagnose in a room, and a stale
        memory list would still offer an object that was just deleted."""
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data, ctype, cache=None):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if cache:
            self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    def _404(self):
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        try:
            if self.path.startswith("/cert.crt") and self.cert:
                # Hand the certificate to the phone so it can be trusted once.
                # This is the same certificate already on the wire, so serving
                # it gives away nothing the TLS handshake does not.
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
                self._send_json(self.app.state())
            elif self.path.startswith("/memories"):
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
                    # filenames are stamped and never rewritten, so the
                    # browser can keep them
                    self._send_bytes(path.read_bytes(), "image/jpeg",
                                     cache="max-age=3600")
                else:
                    self._404()
            elif self.path.startswith("/crop.jpg"):
                jpeg = self.app.crop_jpeg()
                if jpeg:
                    self._send_bytes(jpeg, "image/jpeg")
                else:
                    self._404()
            elif self.path.startswith("/stream"):
                self._stream()
            elif self.path.startswith("/key?k="):
                self.app.keys.put(self.path[-1])
                self.send_response(204)
                self.end_headers()
            elif self.path.startswith("/listen"):
                # phone pressed hold-to-talk: narrate the beat, grab the crop
                self.app.on_listen("t" if self.path.endswith("=t") else "a")
                self.send_response(204)
                self.end_headers()
            else:
                self._404()
        except (BrokenPipeError, ConnectionResetError, ssl.SSLEOFError):
            pass  # tab closed or refreshed mid-stream

    def _stream(self):
        """Send each composed frame once, and always the newest one.

        Pushing on a fixed tick instead sent the same frame repeatedly, which
        is free on Ethernet and fatal on the robot's own hotspot: the radio is
        the bottleneck, and TCP answers oversubscription by queueing, so the
        feed stays smooth and falls further behind the longer you watch.
        Taking the latest shot means a client that falls behind loses frames
        rather than accumulating lag.
        """
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
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

    def do_POST(self):
        """The mutations. Answered with the result rather than a bare 204: the
        memory tab redraws from what the robot says happened, so a delete that
        found nothing cannot leave a card on screen that no longer exists."""
        try:
            if self.path.startswith("/audio"):
                action = "t" if self.path.endswith("=t") else "a"
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                self.send_response(self.app.on_audio(action, body))
                self.end_headers()
            # /forget_view must be routed before /forget: startswith matches both
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
                # capped here, not only in the page: a label is a key that
                # recall reads out loud, and nothing else limits a POST
                label, to = self._query("label"), self._query("to", "")
                to = " ".join((to or "").split())[:60]
                self._send_json(
                    {"label": label, "to": to,
                     "n": self.app.rename(label, to)}
                    if label and to else {"n": 0}, 200 if label and to else 400)
            elif self.path.startswith("/confirm"):
                # tap on the orange "dylan?": teach that crop as that name
                label = self._query("label")
                self._send_json(self.app.confirm(label)
                                if label else {"ok": False},
                                200 if label else 400)
            elif self.path.startswith("/unignore"):
                # the id travels as a string (point ids exceed 2^53); 0 on a
                # malformed request misses harmlessly
                pid = self._int("pid", 0)
                self._send_json({"pid": str(pid),
                                 "ok": self.app.unignore(pid)})
            elif self.path.startswith("/where"):
                # the device moved: change the place stamped on new memories.
                # An empty value clears it.
                self._send_json({"where": self.app.set_where(
                    self._query("to", ""))})
            else:
                self._404()
        except (BrokenPipeError, ConnectionResetError, ssl.SSLEOFError):
            pass
