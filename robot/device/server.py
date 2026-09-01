"""The web server: routes, the MJPEG stream, and the certificate.

Every route is a thin call into `LiveApp` (see live.py). The page itself is
page.html, served whole; the browser polls /state four times a second for
everything that is not video.

HTTP/1.0 is deliberate - each poll is its own connection. Moving to 1.1 would
need an accurate Content-Length on every response or clients hang.
"""
import ipaddress
import json
import socket
import ssl
import time
from datetime import UTC, datetime, timedelta
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


def ensure_cert(ip, root=None):
    """Create or reuse the self-signed certificate required by phone audio.

    The certificate covers the advertised IP, localhost, and the local host
    name. It is a CA certificate so a phone can trust it after installation.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    root = Path(root) if root is not None else (
        Path(__file__).resolve().parents[2] / "cert")
    cert, key, named = root / "cert.pem", root / "key.pem", root / "names.txt"
    hostname = socket.gethostname().split(".")[0] + ".local"
    # localhost stays valid so the laptop view works off the same certificate
    want = ",".join([f"IP:{ip}", "IP:127.0.0.1", "DNS:localhost",
                     f"DNS:{hostname}"])
    if (cert.exists() and key.exists() and named.exists()
            and named.read_text(encoding="utf-8") == want):
        return str(cert), str(key)
    root.mkdir(exist_ok=True)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "qdrant-edge-memory-robot"),
    ])
    now = datetime.now(UTC)
    san = x509.SubjectAlternativeName([
        x509.IPAddress(ipaddress.ip_address(ip)),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.DNSName("localhost"),
        x509.DNSName(hostname),
    ])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=397))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None),
                       critical=True)
        .add_extension(san, critical=False)
        .sign(private_key, hashes.SHA256())
    )
    key.write_bytes(private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    cert.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    named.write_text(want, encoding="utf-8")
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
        """Parse a non-negative integer query value."""
        try:
            return max(0, int(self._query(key, default)))
        except (TypeError, ValueError):
            return default

    def _send_json(self, obj, status=200):
        """Send a complete, non-cached JSON response."""
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
                # Let the phone install the same public certificate TLS serves.
                self.send_response(200)
                self.send_header("Content-Type", "application/x-x509-ca-cert")
                self.send_header("Content-Disposition",
                                 'attachment; filename="qdrant-memory-robot.crt"')
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
                self.app.keys.put(self._query("k", ""))
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
                # tap on the orange "my mug?": teach that crop as that name
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
