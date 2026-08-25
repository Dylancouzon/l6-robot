# Set up a headless Jetson

This setup makes the Jetson start the robot automatically and create its own Wi-Fi network. Run it after the app works normally.

```bash
sudo ./deploy/headless-setup.sh
sudo reboot
```

After the Jetson restarts:

1. Join the network **`l6-robot`**, password **`qdrantedge`**.
2. Open **`https://10.42.0.1:8765`**.
3. Accept the certificate warning, or [install the certificate](phone.md#removing-the-warning-on-your-phone).

Set your own network name and password by passing them to the script:

```bash
sudo SSID=my-robot PSK=my-password ./deploy/headless-setup.sh
```

You can run the setup script again after changing these values.

## What the setup changes

The script starts the robot at boot, restarts it after a crash, and creates the `l6-robot` Wi-Fi hotspot. It also disables the desktop to save memory, enables SSH for maintenance, and downloads the models while the Jetson still has internet access.

## Maintenance

```bash
ssh qdrant@10.42.0.1
journalctl -u l6-robot -f          # the robot's console output
sudo systemctl restart l6-robot    # after editing code or .env
```

If the app stops, the service restarts it after 10 seconds. It may need about 40 seconds to reload the detector. Use the log command above to see the cause.

Set the clock after a cold start. An unplugged robot without internet access does not know the current time, so new memories may have the wrong timestamp. Fix it over SSH:

```bash
sudo timedatectl set-ntp false
sudo timedatectl set-time "2026-08-12 09:30:00"
sudo systemctl restart l6-robot
```

Connecting Ethernet briefly also sets the clock.

To put the robot back on a real network for updates, plug in Ethernet, or `sudo nmcli con up "<your network>"`. The saved profiles are kept, just stopped from autoconnecting. Bring the hotspot back with `sudo nmcli con up l6-hotspot`.

To restore the desktop, run `sudo systemctl set-default graphical.target`. To remove the automatic service, run `sudo systemctl disable --now l6-robot`.
