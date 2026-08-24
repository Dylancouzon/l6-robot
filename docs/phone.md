# Driving It From A Phone

The phone is the robot's whole interface: the screen, the microphone, and the buttons.

```bash
uv run python -m robot.app --host 0.0.0.0
```

The app prints an HTTPS URL. Open it on the phone, accept the certificate warning once, then hold the on-screen buttons to talk. The phone records the audio and uploads it. The Jetson does everything else.

## Why There Is A Certificate Warning

Browsers only allow microphone access in a secure context, so the app serves HTTPS and generates a self-signed certificate on first run. Self-signed means the robot created the certificate itself instead of getting one from a public certificate authority. That is why the browser warns you.

This cannot be coded away. No public certificate authority will sign a certificate for a private address. Reaching one would also need internet access the robot is designed not to need. Accepting the warning once per device is the normal path.

## Removing The Warning On Your Phone

On a phone you use with the robot often, install the certificate as trusted once:

1. Open `https://<address>:8765/cert.crt` and accept the warning one last time to download it. **On iOS this must be done in Safari**, which is the only browser that hands the file to the system as an installable profile.
2. **iOS:** Settings > General > VPN & Device Management, install the profile. Then Settings > General > About > **Certificate Trust Settings** and enable full trust. That last step is easy to miss.
3. **Android:** Settings > Security > Encryption & credentials > Install a certificate > **CA certificate**.

On iOS the trust is system-wide, so every browser on the phone stops warning. Microphone grants start being remembered.

The certificate names the address it was generated for. It is regenerated whenever that address changes, so a phone you trusted on one network will warn again on another. On the [appliance](appliance.md) the address never changes, so the trust is permanent.

## If The Microphone Prompt Appears Every Time

That prompt is the browser's, not the page's. The page asks once per visit and holds the microphone open.

- **iOS Safari:** tap **aA** in the address bar, then Website Settings > Microphone > Allow.
- **Chrome on iOS:** there is no per-site setting. The prompt stops once the certificate is installed and trusted.
- **Android Chrome:** the grant is remembered on its own once the certificate is trusted. Chrome refuses to remember permissions for a site whose certificate it does not trust.

## The Recording Indicator Stays On

The page opens the microphone on your first touch and holds it while the tab lives, so the first word of a press is not clipped by the device opening. That is why the indicator stays lit. Close the tab to release it.

## On A Laptop

The `T` and `A` keys use the laptop's own microphone through `sounddevice`. If the wrong input is selected, set `MIC_DEVICE` in `robot/device/mic.py`. To list devices:

```bash
uv run python -c "import sounddevice; print(sounddevice.query_devices())"
```
