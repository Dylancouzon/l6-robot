# The Headless Appliance

The demo unit has no keyboard, no screen, and no network it can rely on. Set up as an appliance it needs none of them: apply power and it boots into the robot on **its own Wi-Fi network**.

```bash
sudo ./deploy/headless-setup.sh
sudo reboot
```

Then, from a phone anywhere, including a venue with no Wi-Fi at all:

1. Join the network **`l6-robot`**, password **`qdrantedge`**.
2. Open **`https://10.42.0.1:8765`**.
3. Accept the certificate once, or [install it](phone.md#removing-the-warning-on-your-demo-phone). This trust is permanent, because the hotspot address never changes.

Set your own network name and password by passing them to the script:

```bash
sudo SSID=my-robot PSK=my-password ./deploy/headless-setup.sh
```

The script is safe to re-run on a live robot.

## What The Setup Changes

Five things, each reversible on its own:

| Change | Why | Undo |
|---|---|---|
| Installs and enables `l6-robot.service` | Starts at boot, restarts on failure | `sudo systemctl disable --now l6-robot` |
| Adds an `l6-hotspot` profile on 5 GHz channel 44, and stops saved networks autoconnecting | One radio cannot be an access point and a client at once, and the robot must work where there is no network | `sudo nmcli con delete l6-hotspot`, then re-enable autoconnect on your own network |
| `systemctl set-default multi-user.target` | Frees roughly 1.5 GB of desktop on an 8 GB board | `sudo systemctl set-default graphical.target` |
| Generates ssh host keys and starts `sshd` | The only way into a box with no peripherals | |
| Fills the model cache under `$HOME` | FastEmbed defaults to `/tmp`, which is pruned at 30 days. A robot with no internet would boot into a download that never finishes | |

## Turning It On And Off

There is no power button and the Jetson does not need one: it turns on as soon as the DC supply is connected. Plugging the case in is the on switch.

**Pulling the plug is the off switch, and it is safe for the memories.** Every teach writes one point and flushes it to disk immediately, so there is no buffered state to lose. For a graceful shutdown anyway, `ssh qdrant@10.42.0.1 sudo poweroff`.

## Two Things That Slow A Demo

**Your phone keeps its cellular service.** Joining the robot only takes over the phone's Wi-Fi, and both iOS and Android keep internet traffic on mobile data once they notice the robot's network has no internet path. iOS shows "No Internet Connection" under the network name and works fine. On Android, turn off "Switch to mobile data automatically" for that phone if it drops the association.

**Do not leave a laptop joined to the hotspot.** When the robot has Ethernet, its hotspot shares that connection, so a laptop routes every background sync and update through the same radio that carries the video feed. That is the most common cause of a demo that looks slow, and nothing on the robot is at fault. Disconnect the laptop and it clears immediately. Phones mostly dodge this by keeping their traffic on cellular.

## Set The Clock Before Filming

Recall reads times out loud, and a robot that has been unplugged with no internet does not know what time it is. There is no time server on its own hotspot, so a unit switched on cold at a venue stamps new memories with the time it was last packed away.

Either set it over ssh when you set up for the day:

```bash
sudo timedatectl set-ntp false
sudo timedatectl set-time "2026-08-12 09:30:00"
sudo systemctl restart l6-robot
```

Or fit a coin cell to the RTC backup connector on the carrier board, which keeps the clock running with the power off. Plugging in Ethernet for a minute also fixes it.

## Maintenance

```bash
ssh qdrant@10.42.0.1
journalctl -u l6-robot -f          # the robot's console output
sudo systemctl restart l6-robot    # after editing code or .env
```

Two behaviours worth knowing before you debug them:

- **A crash is invisible and self-healing.** The service restarts 10 seconds later and takes about 40 seconds to reload the detector, so an unplugged camera looks like a robot that is simply slow to come back. `journalctl -u l6-robot` has the reason.
- **The service runs with `--watchdog 30`.** A USB camera that wedges inside a driver call cannot be noticed by the thread stuck in it, so the app exits if no frame arrives for 30 seconds and lets the service manager restart it. A restart with no error above it was this.

To put the robot back on a real network for updates, plug in Ethernet, or `sudo nmcli con up "<your network>"`. The saved profiles are kept, just stopped from autoconnecting. Bring the hotspot back with `sudo nmcli con up l6-hotspot`.

## Running On A Jetson Orin Nano

The same run commands work on the 8 GB board. Four things are worth knowing:

- **Torch prints a compute-capability warning.** It is cosmetic. Orin is `sm_87` and runs the wheel's `sm_80` kernels, and CUDA is genuinely in use: the detector runs at roughly 165 ms per frame against seconds per frame on CPU.
- **Only the CLIP vision encoder loads at startup.** Speech and text encoders load the first time you teach or ask, which is what keeps the board out of swap. Expect roughly 2.5 GB after startup.
- **Keep the stock 15 W power mode.** The bottleneck is memory and the USB camera, not the GPU, so the faster power modes buy nothing.
- **"System throttled due to Over-current" can pop up repeatedly.** In this demo it does not mean the robot is too slow or underpowered. `OC3` counts instantaneous spikes caught by a hardware comparator, far too fast for the power sensor to sample, which is why `tegrastats` shows nothing near the limit. Sustained draw here is about 1.9 A against a 4.05 A limit, and detector latency stays flat. The appliance setup removes the desktop applet that shows the popup.

## The Feed Is The Bottleneck

The live view is MJPEG, a stream of JPEG images sent as video. At 10 fps it is well over 10 Mbps over the robot's own hotspot, which is the narrowest part of the chain. Exactly how much depends on the scene, not the code: a cluttered desk measured 210 KB per frame here, a tidier one 183 KB.

That is why the hotspot runs on 5 GHz. A 2.4 GHz access point carries 15 to 25 Mbps in an empty room and much less in a hall full of phones. When the radio runs out, the failure is not a dropped frame: TCP queues what it cannot send, so the feed arrives smooth but seconds behind the room, and the delay grows the longer you watch. Pressing TEACH on an object you can no longer see is the symptom that ruins a demo. The trade is range, so keep the phone in the same room as the robot.

If the feed lags, measure before changing settings:

```bash
iw dev wlP1p1s0 info    # which band and channel the access point is really on
timeout 5 curl -sk https://127.0.0.1:8765/stream -o /dev/null -w '%{size_download}\n'
```

The second command measures what the robot wants to send with the radio out of the picture. Divide by 5 for bytes per second. If that number is comfortable and the phone still lags, the problem is the link, not the robot.

`STREAM_QUALITY` in `robot/device/live.py` is the one number to turn, and only against a measurement of your own scene: JPEG size depends far more on what the camera is pointed at than on anything in the code.
