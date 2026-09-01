# Set Up a Headless Jetson

This setup makes the Jetson start the robot automatically and create its own Wi-Fi network. Run it after the app works normally.

```bash
sudo ./deploy/headless-setup.sh
sudo reboot
```

Run this from the Jetson's own console, over Ethernet, or over the USB-C connection. The last step creates the hotspot, which takes the Wi-Fi radio away from any network the Jetson is currently joined to. An SSH session running over Wi-Fi ends there, partway through the setup.

After the Jetson restarts:

1. Join the network **`qdrant-memory`**, password **`qdrantedge`**.
2. Open **`https://10.42.0.1:8765`**.
3. Accept the certificate warning, or [install the certificate](phone.md#remove-the-warning-from-your-phone).

Set your own network name and password by passing them to the script:

```bash
sudo SSID=my-robot PSK=my-password ./deploy/headless-setup.sh
```

You can run the setup script again after changing these values.

## What the Setup Changes

The script starts the robot at boot, restarts it after a crash, and creates the `qdrant-memory` Wi-Fi hotspot. It also disables the desktop to save memory, enables SSH for maintenance, keeps the clock across power cuts, and downloads the models while the Jetson still has internet access.

Run the script with `sudo` from the same account that owns the repository. It detects that account, the repository path, and the `uv` executable when it installs the service. To install for a different account, set `ROBOT_USER` and, if needed, `UV_BIN`.

## Maintenance

```bash
ssh your-user@10.42.0.1
journalctl -u memory-robot -f          # the robot's console output
sudo systemctl restart memory-robot    # after editing code or .env
```

If the app stops, the service restarts it after 10 seconds. It may need about 40 seconds to reload the detector. Use the log command above to see the cause.

## The Clock

The Jetson keeps no time while it is unplugged, and a robot on its own hotspot has no internet access to ask for it. Left alone, a cold start comes up at 1 January 1970 and stamps new memories with that date, which sorts them behind every real memory and hides them from recall.

The `memory-robot-clock` service handles this. It writes the time to a file every minute and puts it back at the next boot, so a power cut costs at most a minute. It only ever moves the clock forward, and it rewrites that file from the running clock every minute, so a correction from Ethernet or from you is picked up within a minute and carried forward from there.

Give it a true time once, and it carries that time forward:

```bash
sudo timedatectl set-ntp false
sudo timedatectl set-time "2026-08-12 09:30:00"
sudo systemctl restart memory-robot
```

Restart the robot after any manual change. It picks the starting number for memory IDs from the clock when it opens the shard, so a running robot keeps using the old time until it restarts.

Connecting Ethernet briefly also sets the clock. To see what the robot believes, run `timedatectl`.

Because the service only moves the clock forward, a time set far into the future by mistake sticks: the saved file is then always ahead of the correct time. Delete it and set the clock again.

```bash
sudo systemctl stop memory-robot-clock
sudo rm -f /var/lib/clock-keeper/stamp
sudo timedatectl set-time "2026-08-12 09:30:00"
sudo systemctl start memory-robot-clock
sudo systemctl restart memory-robot
```

A CR2032 coin cell on header **J3** lets the board keep time on its own, which removes the problem rather than working around it. `J13` beside it is the fan header.

To put the robot back on a real network for updates, plug in Ethernet, or `sudo nmcli con up "<your network>"`. The setup keeps saved profiles but disables their automatic connection. Bring the hotspot back with `sudo nmcli con up memory-robot-hotspot`.

To restore the desktop, run `sudo systemctl set-default graphical.target`. To remove the automatic services, run `sudo systemctl disable --now memory-robot memory-robot-clock`.
