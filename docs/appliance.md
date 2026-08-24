# The Headless Appliance

The robot needs no keyboard, no screen, and no network. Set it up as an appliance: apply power and it boots into the robot on its own Wi-Fi network.

```bash
sudo ./deploy/headless-setup.sh
sudo reboot
```

Then use a phone anywhere, including a venue with no Wi-Fi at all:

1. Join the network **`l6-robot`**, password **`qdrantedge`**.
2. Open **`https://10.42.0.1:8765`**.
3. Accept the certificate once, or [install it](phone.md#removing-the-warning-on-your-phone). This trust is permanent, because the hotspot address never changes.

Set your own network name and password by passing them to the script:

```bash
sudo SSID=my-robot PSK=my-password ./deploy/headless-setup.sh
```

You can re-run the script on a live robot.

## What The Setup Changes

The setup changes five things. Each one is reversible on its own.

| Change | Why | Undo |
|---|---|---|
| Installs and enables `l6-robot.service`, a systemd service that runs the robot in the background | Starts at boot, restarts on failure | `sudo systemctl disable --now l6-robot` |
| Adds an `l6-hotspot` profile on 5 GHz channel 44, and stops saved networks autoconnecting | One radio cannot be an access point and a client at once. The robot must work where there is no network | `sudo nmcli con delete l6-hotspot`, then re-enable autoconnect on your own network |
| `systemctl set-default multi-user.target` | Frees roughly 1.5 GB of desktop on an 8 GB board | `sudo systemctl set-default graphical.target` |
| Generates ssh host keys and starts `sshd` | The only way into a box with no peripherals | |
| Fills the model cache under `$HOME` | FastEmbed defaults to `/tmp`, which is pruned at 30 days. A robot with no internet would boot into a download that never finishes | |

## Turning It On And Off

There is no power button, and the Jetson does not need one. It turns on as soon as the DC supply is connected. Plugging the case in is the on switch.

**Pulling the plug is the off switch, and it is safe for the memories.** Every teach writes one point and flushes it to disk immediately, so there is no buffered state to lose. For a graceful shutdown anyway, run `ssh qdrant@10.42.0.1 sudo poweroff`.

## Maintenance

```bash
ssh qdrant@10.42.0.1
journalctl -u l6-robot -f          # the robot's console output
sudo systemctl restart l6-robot    # after editing code or .env
```

Three behaviours worth knowing before you debug them:

- **A crash is invisible and self-healing.** The service restarts 10 seconds later and takes about 40 seconds to reload the detector. An unplugged camera can look like a robot that is simply slow to come back. `journalctl -u l6-robot` has the reason.
- **The service runs with `--watchdog 30`.** A USB camera that wedges inside a driver call cannot be noticed by the thread stuck in it. The app exits if no frame arrives for 30 seconds and lets the service manager restart it. A restart with no error above it was this.
- **Do not leave a laptop joined to the hotspot.** When the robot has Ethernet, its hotspot shares that connection, and a laptop routes every background sync through the same radio that carries the video feed. If the feed lags, disconnect the laptop first. Phones mostly dodge this by keeping their traffic on cellular.

**Set the clock after a cold start.** Recall reads times out loud, and an unplugged robot with no internet does not know what time it is, so it stamps new memories with the time it was last powered. Fix it over ssh:

```bash
sudo timedatectl set-ntp false
sudo timedatectl set-time "2026-08-12 09:30:00"
sudo systemctl restart l6-robot
```

Plugging in Ethernet for a minute also fixes it, and a coin cell on the carrier board's RTC backup connector keeps the clock running with the power off.

To put the robot back on a real network for updates, plug in Ethernet, or `sudo nmcli con up "<your network>"`. The saved profiles are kept, just stopped from autoconnecting. Bring the hotspot back with `sudo nmcli con up l6-hotspot`.

## Running On A Jetson Orin Nano

The same run commands work on the 8 GB board. Four things are worth knowing:

- **Torch prints a compute-capability warning.** It is cosmetic. Orin is `sm_87` and runs the wheel's `sm_80` kernels. CUDA is genuinely in use: the detector runs at roughly 165 ms per frame against seconds per frame on CPU.
- **Only the CLIP vision encoder loads at startup.** Speech and text encoders load the first time you teach or ask. That keeps the board out of swap. Expect roughly 2.5 GB after startup.
- **Keep the stock 15 W power mode.** The bottleneck is memory and the USB camera, not the GPU. The faster power modes buy nothing.
- **"System throttled due to Over-current" popups are expected.** `OC3` counts instantaneous current spikes far too fast for the power sensor to sample, and detector latency stays flat while they tick. The appliance setup removes the desktop applet that shows the popup.
