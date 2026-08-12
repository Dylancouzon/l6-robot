# Getting Into The Robot

The demo unit runs headless: no keyboard, no screen, no peripherals, and often no
network it can rely on. This is how you get a shell on it anyway, how you work on
it once you are in, and how you move it between Wi-Fi networks.

Everything here was verified on the unit itself. Where something is an
expectation rather than a measurement, it says so.

The robot's account is `qdrant`, and the repo lives at
`~/Documents/github/l6-robot`.

## Three Ways In

| Route | Command | Works when | Costs |
|---|---|---|---|
| **Your LAN** | `ssh qdrant@qdrant-robot.local` | The robot is on a desk, wired to your router. You can be on the router's Wi-Fi — you do not need a cable | Nothing. This is the best route: the robot gets internet, *and* its hotspot shares that internet with your phone |
| **Robot's own hotspot** | `ssh qdrant@10.42.0.1` | Anywhere. Needs no infrastructure at all | The robot has no internet, so Claude Code will not run — see [Running Claude Code On The Robot](#running-claude-code-on-the-robot) |
| **USB-C debug port** | `ssh qdrant@192.168.55.1` | Always — no network, no router, no Wi-Fi | Needs physical access to the port and a USB-C cable |

If all three fail, the same USB-C cable also carries a **serial console**, which
works even when networking is completely broken. See
[When You Are Locked Out](#when-you-are-locked-out).

`sshd` is enabled and its host keys exist. It listens on every interface, so one
running daemon covers all three routes at once.

### Finding it on your LAN without a screen

The normal desk setup is the robot **wired to the router** and you on the
router's Wi-Fi. They are the same network, so nothing special is needed. Note the
robot's own radio is an access point, not a client, so the cable is its only way
onto your LAN and its only route to the internet.

You do not need to know the robot's DHCP address — `avahi-daemon` runs, so the
name resolves from any machine on the same LAN:

```bash
ssh qdrant@qdrant-robot.local
```

Two things that can bite:

- **The name may resolve to IPv6 first.** On this network `qdrant-robot.local`
  returns only IPv6 addresses, and the wired IPv4 address is a DHCP lease that
  can change. If the name behaves oddly, use the address:
  `ip -4 addr show enP8p1s0` on the robot, currently `192.168.1.143`.
- **Guest networks and AP isolation.** Some routers stop wireless clients talking
  to wired ones, which makes the robot invisible from your laptop for a reason
  that has nothing to do with the robot. Use the USB-C port instead of debugging
  the router.

### Why there is no direct-Ethernet route

Running a cable straight from your laptop to the robot is deliberately not listed,
because the USB-C port does the same job better:

- **The robot has one Ethernet port.** A direct cable spends it, and that port is
  the robot's only internet — and therefore the only thing that lets Claude Code
  run on the box. USB-C is additive: keep the router cable *and* have the debug
  link at the same time.
- **No DHCP server on either end**, so both sides fall back to link-local after a
  timeout and you are guessing at `169.254.x.x`. The USB-C link is always
  `192.168.55.1`, with the robot handing your laptop an address.
- **The same USB-C cable also carries the serial console**, so it degrades to a
  working route when networking is broken. A direct Ethernet cable does not.

The only thing a direct cable wins is bulk transfer speed, and the router route
already gives you that at gigabit.

## The USB-C Debug Port

This is the route that always works, and the one worth practising before you
need it. The Jetson's USB-C port runs in **USB device mode**: plug a USB-C cable
from your laptop into the robot and the robot appears as a USB network adapter
*and* as a serial port.

```bash
ssh qdrant@192.168.55.1
```

The robot is always `192.168.55.1` on that link, and it runs a small DHCP server
that hands your laptop `192.168.55.100`. Nothing about this depends on Wi-Fi,
Ethernet, DNS, or the hotspot, which is why it is the fallback: it still works if
you have broken the network configuration entirely.

Verified on this unit: `nv-l4t-usb-device-mode` is enabled and active, the
`l4tbr0` bridge holds `192.168.55.1`, and the link sits at `NO-CARRIER` until
something is plugged in.

Do not confuse this with recovery mode. Nothing here reflashes anything.

## Running Claude Code On The Robot

Claude Code is installed on the robot (`~/.local/bin/claude`). So:

```bash
ssh qdrant@10.42.0.1
cd ~/Documents/github/l6-robot
tmux new -s work        # see below — do not skip this
claude
```

**It needs internet, and the hotspot does not provide any.** This was tested: in
a network namespace with nothing but loopback, `claude` fails immediately. The
hotspot is a LAN, not an uplink. Plain shell work over the hotspot is fine —
`ssh`, `git` against a local clone, editing, running the robot — it is only
Claude Code and other network tools that need a real connection.

So when you want to work with Claude on the box, pick one:

1. **Wire it to your router.** Best option by a distance. The robot gets
   internet, and its hotspot *shares* that internet, so your phone gets real
   internet on `l6-robot` too. You stay on Wi-Fi; only the robot needs the cable.
2. **Put the Wi-Fi back into client mode** — see
   [Moving It Between Wi-Fi Networks](#moving-it-between-wi-fi-networks). Expect
   this to take the hotspot down, so do it over Ethernet or the USB-C port,
   never over the hotspot you are about to switch off.
3. **Work from your laptop** on a clone and `git pull` on the robot.

### Always use tmux

`tmux` is installed for one reason: a dropped Wi-Fi connection otherwise kills
whatever Claude Code was doing, mid-edit.

```bash
tmux new -s work        # start
# Ctrl-b then d         # detach, leaving it running
tmux attach -t work     # come back, from any of the three routes
```

Detached sessions survive a dropped connection, a closed laptop, and switching
from hotspot to Ethernet mid-session. They do **not** survive a reboot.

## Working On The Robot

The service holds two things exclusively, so stop it before running the app by
hand. Both were tested:

- **The camera.** A second process gets `isOpened() == False`.
- **The shard.** A second writer is refused outright with a WAL error — good
  news, because it means you cannot corrupt your memories by accidentally
  running two robots.

```bash
sudo systemctl stop l6-robot
uv run python -m robot.app --host 0.0.0.0 --advertise 10.42.0.1
# Ctrl-C when done, then hand the camera back:
sudo systemctl start l6-robot
```

Day-to-day:

```bash
journalctl -u l6-robot -f          # the robot's console output, live
sudo systemctl restart l6-robot    # after editing code or .env
systemctl status l6-robot          # is it running, and why did it restart
```

After editing `deploy/l6-robot.service`, re-run `sudo ./deploy/headless-setup.sh`
and then restart. The script is idempotent and will not drop phones that are
already connected to the hotspot.

## Moving It Between Wi-Fi Networks

The robot's radio is an access point by default: profile `l6-hotspot`, at
`connection.autoconnect-priority 100`, with every other saved Wi-Fi profile
deliberately **parked** (`autoconnect no`). Parking matters — a saved network
that autoconnects first would win the radio at boot and the robot would come up
with no hotspot and no obvious way in.

**Do all of this over Ethernet or the USB-C port.** Joining a network as a client
should be expected to take the hotspot down, which would kill an ssh session
running over that hotspot. (One radio, two roles: this chip's driver advertises
no interface combinations, so treat concurrent AP + client as unsupported rather
than something to rely on.)

### See what is around, and what is already saved

Scanning is safe while the hotspot is running — tested, and the access point
stayed up throughout:

```bash
sudo nmcli device wifi list --rescan yes     # what is in range
nmcli con show                               # what is already saved
```

### A network it already knows

```bash
sudo nmcli con up "FBI surveillance van #4"    # or whichever profile
```

### A new network, at a venue or a hotel

```bash
sudo nmcli device wifi connect "<SSID>" password "<password>" name "<profile-name>"
sudo nmcli con modify "<profile-name>" connection.autoconnect no    # park it
```

That second line is not optional. Without it the new profile will fight the
hotspot for the radio on the next boot.

### A hidden network

```bash
sudo nmcli con add type wifi con-name "<profile-name>" ifname wlP1p1s0 \
  ssid "<SSID>" 802-11-wireless.hidden yes connection.autoconnect no
sudo nmcli con modify "<profile-name>" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "<password>"
sudo nmcli con up "<profile-name>"
```

### Back to being a robot

```bash
sudo nmcli con up l6-hotspot
```

This also happens by itself on the next reboot, because the hotspot is the only
Wi-Fi profile left with `autoconnect yes`.

### Networks that will not work

A captive portal — hotel or conference Wi-Fi with a sign-in page — cannot be
accepted from a headless box without a browser. If the robot needs internet in
that situation, use Ethernet or your phone's hotspot instead.

## When You Are Locked Out

In order, from least to most disruptive:

1. **Try the other routes.** They are genuinely independent: hotspot, Ethernet,
   and USB-C fail for different reasons.
2. **Serial console over the same USB-C cable.** A login prompt on the robot's
   `ttyGS0`, verified running. From a laptop:

   ```bash
   screen /dev/ttyACM0 115200          # Linux
   screen /dev/tty.usbmodem* 115200    # macOS
   ```

   This is a real console, not a network service, so it works when networking is
   broken, when the hotspot never came up, and when `sshd` is dead. Leave with
   `Ctrl-a` then `k`.
3. **Monitor and keyboard.** Keep them within reach until the unit has completed
   at least one successful cold boot in its case.

To undo the headless configuration entirely and get the desktop back:

```bash
sudo systemctl set-default graphical.target
sudo systemctl disable --now l6-robot
sudo reboot
```

## Quick Reference

```bash
# get in
ssh qdrant@qdrant-robot.local        # your LAN (robot wired to the router, you on wi-fi)
ssh qdrant@10.42.0.1                 # the robot's own hotspot, works anywhere
ssh qdrant@192.168.55.1              # USB-C cable, works with no network at all
screen /dev/ttyACM0 115200           # USB-C serial console, works when nothing else does

# survive a dropped connection
tmux new -s work / Ctrl-b d / tmux attach -t work

# the robot
journalctl -u l6-robot -f
sudo systemctl restart l6-robot
sudo systemctl stop l6-robot         # before running the app by hand

# wi-fi
sudo nmcli device wifi list --rescan yes
sudo nmcli con up "<saved network>"  # expect the hotspot to drop
sudo nmcli con up l6-hotspot         # back to being a robot
```

Operator-facing setup, the phone demo, and what the appliance configuration
changes on the machine are in [README.md](README.md) under "Headless Appliance".
