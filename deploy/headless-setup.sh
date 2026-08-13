#!/usr/bin/env bash
# Turn this machine into the robot appliance: no desktop, no peripherals, no
# venue Wi-Fi. Idempotent — safe to re-run after editing anything here or in
# deploy/l6-robot.service.
#
#   sudo ./deploy/headless-setup.sh
#
# Run it from the console or over Ethernet. The Wi-Fi step converts the radio
# into an access point, which drops any ssh session arriving over Wi-Fi — so it
# is deliberately the LAST thing this script does, after the parts that need a
# working internet connection.
#
# Afterwards: reboot, join the Wi-Fi network below from a phone, open the URL.
# Undoing it is four commands, listed in the README under "Headless Appliance".
set -euo pipefail

SSID="${SSID:-l6-robot}"
PSK="${PSK:-qdrantedge}"          # WPA2 needs 8+ characters
# Fixed, deliberately not a knob: the unit's --advertise names this address in
# the certificate, and the two disagreeing is the name-mismatch warning the
# whole headless path exists to avoid.
AP_IP="10.42.0.1"
# Also not a knob, and named once so the modify below and the already-up check
# further down cannot disagree about which channel the radio should be on.
# Reasoning for 5 GHz and for this channel is at the Wi-Fi step.
BAND="a"
CHANNEL="44"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT="$REPO/deploy/l6-robot.service"
# Read the account out of the unit rather than taking it as a knob: the unit
# also pins WorkingDirectory to that home, and the two disagreeing means the
# model cache gets warmed for a user the robot does not run as.
USER_NAME="$(awk -F= '/^User=/{print $2}' "$UNIT")"

[[ $EUID -eq 0 ]] || { echo "run me with sudo" >&2; exit 1; }
[[ -n "$USER_NAME" ]] || { echo "no User= in $UNIT" >&2; exit 1; }
say() { printf '\n== %s\n' "$*"; }

say "sshd — the only way into a box with no peripherals"
# A missing host key leaves sshd enabled but dead, which you discover at the
# worst possible moment. ssh-keygen -A only creates what is absent.
ssh-keygen -A
systemctl reset-failed ssh.service ssh.socket 2>/dev/null || true
systemctl enable --now ssh
if systemctl is-active --quiet ssh; then echo "ssh: listening"
else echo "ssh: NOT running — check 'journalctl -u ssh'" >&2; fi

say "model cache — 1.1 GB the robot cannot re-download once it is offline"
# Before the Wi-Fi step, because after it this machine has no internet unless
# Ethernet is plugged in. FastEmbed's default cache lives in /tmp, which
# systemd-tmpfiles prunes; the app pins it under $HOME (robot/models.py), so
# fill it now — after this the robot boots with no network at all.
sudo -u "$USER_NAME" -H env PATH="/home/$USER_NAME/.local/bin:$PATH" \
  sh -c "cd '$REPO' && uv run --no-sync python -c '
from robot import models
models.warm_up(lambda n: print(\"  cached\", n))
models._text_model(); models._asr_model()
print(\"  cached speech and text encoders\")'"

say "service — start on boot, restart on failure, live whenever the box is on"
install -m 644 "$UNIT" /etc/systemd/system/l6-robot.service
systemctl daemon-reload
systemctl enable l6-robot.service
# Deliberately not restarted here: a re-run must not take a live robot down
# mid-demo. A changed unit waits until someone asks for it.
echo "enabled (if the unit changed: systemctl restart l6-robot)"

say "desktop — off, which frees about 1.5 GB on an 8 GB board"
# Takes effect at the next boot; it deliberately does not kill your session now.
systemctl set-default multi-user.target
# The over-current popup is a desktop applet, so this retires it too.

say "Wi-Fi — the robot serves its own network, so it works anywhere"
wifi_dev="$(nmcli -t -f DEVICE,TYPE device status | awk -F: '$2=="wifi"{print $1; exit}')"
[[ -n "$wifi_dev" ]] || { echo "no wifi device found" >&2; exit 1; }
if ! nmcli -t -f NAME con show | grep -qx l6-hotspot; then
  nmcli con add type wifi ifname "$wifi_dev" con-name l6-hotspot ssid "$SSID" >/dev/null
fi
# 5 GHz, not 2.4. The MJPEG feed is the whole demo and it wants ~13 Mbps; a
# 2.4 GHz 20 MHz AP delivers 15-25 Mbps in an empty room and much less in a hall
# full of phones, so the link becomes the bottleneck, TCP queues rather than
# drops, and the feed arrives as a slideshow seconds behind the room. Channel 44
# is non-DFS (no radar wait before it may transmit) and clear of the 149-165
# block most venue kit crowds into. The trade is range: 5 GHz carries less far
# and through less, so keep the phone in the same room as the robot.
# ssid is set here too, not only on create: without it, re-running with a new
# SSID= would change the password and print the new name while the radio kept
# broadcasting the old one.
nmcli con modify l6-hotspot \
  802-11-wireless.ssid "$SSID" \
  802-11-wireless.mode ap \
  802-11-wireless.band "$BAND" 802-11-wireless.channel "$CHANNEL" \
  802-11-wireless.powersave 2 \
  ipv4.method shared ipv4.addresses "$AP_IP/24" ipv6.method ignore \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$PSK" \
  connection.autoconnect yes connection.autoconnect-priority 100
# One radio cannot be an access point and a client at once, and a saved network
# that autoconnects first wins the device. Keep the profiles (so `nmcli con up
# <name>` still gets you online) but stop them starting themselves. Iterate by
# UUID: a profile name may contain a colon, which would break `IFS=:` parsing
# and silently leave that profile free to grab the radio at boot.
while IFS=: read -r uuid type; do
  [[ "$type" == "802-11-wireless" ]] || continue
  name="$(nmcli -g connection.id con show "$uuid")"
  [[ "$name" != "l6-hotspot" ]] || continue
  echo "  parking saved network: $name (autoconnect off)"
  nmcli con modify "$uuid" connection.autoconnect no
done < <(nmcli -t -f UUID,TYPE con show)
# Don't bounce an access point that is already serving: re-running this script
# should not drop the phones currently connected to it.
if nmcli -t -f NAME con show --active | grep -qx l6-hotspot; then
  # ...but say so when the profile no longer matches the air. A changed band or
  # channel only reaches the radio on a bounce, and the same silent-mismatch
  # trap as the ssid above applies: printing the new channel while the old one
  # broadcasts sends you debugging the wrong layer.
  # Both the name and the channel: a re-run with a new SSID= writes the profile
  # and would otherwise print the new name while the radio still broadcasts the
  # old one, sending the room to a network that does not exist.
  info="$(iw dev "$wifi_dev" info 2>/dev/null || true)"
  live_ch="$(awk '$1=="channel"{print $2}' <<<"$info")"
  live_ssid="$(sed -n 's/^[[:space:]]*ssid //p' <<<"$info")"
  if [[ -n "$live_ch" && "$live_ch" != "$CHANNEL" ]] ||
     [[ -n "$live_ssid" && "$live_ssid" != "$SSID" ]]; then
    echo "hotspot is up as \"$live_ssid\" on channel $live_ch;"
    echo "  the profile now says \"$SSID\" on channel $CHANNEL"
    echo "  to apply it (drops connected clients): nmcli con up l6-hotspot"
  else
    echo "hotspot already up"
  fi
else
  nmcli con up l6-hotspot >/dev/null
fi
echo "hotspot: $SSID / $PSK at $AP_IP"

cat <<EOF

Done. Reboot, then from a phone:
  1. join Wi-Fi "$SSID", password "$PSK"
  2. open https://$AP_IP:8765
  3. accept the certificate once (or install it from https://$AP_IP:8765/cert.crt
     to stop being asked — see the README)

Maintenance, from a laptop on the same Wi-Fi:
  ssh $USER_NAME@$AP_IP
  journalctl -u l6-robot -f
EOF
