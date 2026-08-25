# Use a phone

The phone is the robot's whole interface: the screen, the microphone, and the buttons.

```bash
uv run python -m robot.app --host 0.0.0.0
```

The app prints an HTTPS URL. Open it on the phone, accept the certificate warning, then hold the on-screen buttons to talk.

## Why the browser shows a certificate warning

Browsers only allow microphone access over a secure connection. The app therefore uses HTTPS and creates its own certificate on the first run. Because a public certificate authority did not issue it, the browser shows a warning.

## Remove the warning from your phone

On a phone you use with the robot often, install the certificate as trusted once:

1. Open `https://<address>:8765/cert.crt` and accept the warning one last time to download it. On iOS, use Safari. Other browsers do not pass the file to the system as an installable profile.
2. On iOS, open Settings > General > VPN & Device Management and install the profile. Then open Settings > General > About > Certificate Trust Settings and enable full trust.
3. On Android, open Settings > Security > Encryption & credentials > Install a certificate > CA certificate.

On iOS the trust is system-wide, so every browser on the phone stops warning. Microphone grants start being remembered.

The certificate names the address it was generated for. It is regenerated whenever that address changes, so a phone you trusted on one network will warn again on another. On the [appliance](appliance.md) the address never changes, so the trust is permanent.

## If the microphone prompt appears every time

That prompt is the browser's, not the page's. The page asks once per visit and holds the microphone open.

- In iOS Safari, tap **aA** in the address bar, then Website Settings > Microphone > Allow.
- Chrome on iOS has no setting for an individual site. The prompt stops once the certificate is installed and trusted.
- Android Chrome remembers the permission once the certificate is trusted.

## Why the recording indicator stays on

The page opens the microphone on your first touch and holds it while the tab lives, so the first word of a press is not clipped by the device opening. That is why the indicator stays lit. Close the tab to release it.

## On a laptop

The `T` and `A` keys use the laptop's own microphone through `sounddevice`. If the wrong input is selected, set `MIC_DEVICE` in `robot/device/mic.py`. To list devices:

```bash
uv run python -c "import sounddevice; print(sounddevice.query_devices())"
```
