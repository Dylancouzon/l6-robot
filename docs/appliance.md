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

The script starts the robot at boot, restarts it after a crash, and creates the `qdrant-memory` Wi-Fi hotspot. It also disables the desktop to save memory, enables SSH for maintenance, and downloads the models while the Jetson still has internet access.

Run the script with `sudo` from the same account that owns the repository. It detects that account, the repository path, and the `uv` executable when it installs the service. To install for a different account, set `ROBOT_USER` and, if needed, `UV_BIN`.

## Maintenance

```bash
ssh your-user@10.42.0.1
journalctl -u memory-robot -f          # the robot's console output
sudo systemctl restart memory-robot    # after editing code or .env
```

If the app stops, the service restarts it after 10 seconds. It may need about 40 seconds to reload the detector. Use the log command above to see the cause.

Set the clock after a cold start. An unplugged robot without internet access does not know the current time, so new memories may have the wrong timestamp. Fix it over SSH:

```bash
sudo timedatectl set-ntp false
sudo timedatectl set-time "2026-08-12 09:30:00"
sudo systemctl restart memory-robot
```

Connecting Ethernet briefly also sets the clock.

To put the robot back on a real network for updates, plug in Ethernet, or `sudo nmcli con up "<your network>"`. The setup keeps saved profiles but disables their automatic connection. Bring the hotspot back with `sudo nmcli con up memory-robot-hotspot`.

To restore the desktop, run `sudo systemctl set-default graphical.target`. To remove the automatic service, run `sudo systemctl disable --now memory-robot`.
