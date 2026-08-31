#!/usr/bin/env bash
# Configure the Jetson as a headless robot with its own Wi-Fi network.
#
#   sudo ./deploy/headless-setup.sh
#
# Run from the console or Ethernet. The final Wi-Fi step drops Wi-Fi SSH.
# Maintenance and removal commands are in docs/appliance.md.
set -euo pipefail

SSID="${SSID:-qdrant-memory}"
PSK="${PSK:-qdrantedge}"          # WPA2 needs 8+ characters
SERVICE_NAME="memory-robot"
HOTSPOT_NAME="memory-robot-hotspot"
# This must match the address passed to --advertise in the service.
AP_IP="10.42.0.1"
BAND="a"
CHANNEL="44"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT="$REPO/deploy/memory-robot.service"
USER_NAME="${ROBOT_USER:-${SUDO_USER:-}}"

[[ $EUID -eq 0 ]] || { echo "run me with sudo" >&2; exit 1; }
if [[ -z "$USER_NAME" || "$USER_NAME" == "root" ]]; then
  echo "run with sudo from the robot account, or set ROBOT_USER" >&2
  exit 1
fi
id "$USER_NAME" >/dev/null 2>&1 || {
  echo "user does not exist: $USER_NAME" >&2
  exit 1
}
USER_HOME="$(getent passwd "$USER_NAME" | cut -d: -f6)"
[[ -n "$USER_HOME" ]] || { echo "no home found for $USER_NAME" >&2; exit 1; }
if [[ -z "${UV_BIN:-}" ]]; then
  for candidate in "$USER_HOME/.local/bin/uv" /usr/local/bin/uv /usr/bin/uv; do
    if [[ -x "$candidate" ]]; then
      UV_BIN="$candidate"
      break
    fi
  done
fi
[[ -x "${UV_BIN:-}" ]] || {
  echo "uv not found; install it for $USER_NAME or set UV_BIN" >&2
  exit 1
}
say() { printf '\n== %s\n' "$*"; }

say "sshd: access to the headless robot"
# Create any missing host keys before enabling SSH.
ssh-keygen -A
systemctl reset-failed ssh.service ssh.socket 2>/dev/null || true
systemctl enable --now ssh
if systemctl is-active --quiet ssh; then echo "ssh: listening"
else echo "ssh: NOT running; check 'journalctl -u ssh'" >&2; fi

say "model cache: download before the robot goes offline"
# The app stores models under the user's home directory, not temporary storage.
(
cd "$REPO"
sudo -u "$USER_NAME" -H env HOME="$USER_HOME" "$UV_BIN" \
  run --no-sync python -c '
from robot.brain import models
models.warm_up(lambda n: print("  cached", n))
models._text_model(); models._asr_model()
print("  cached speech and text encoders")'
)

say "service: start on boot and restart on failure"
escape_sed() { sed 's/[&|\\]/\\&/g' <<<"$1"; }
rendered_unit="$(mktemp)"
trap 'rm -f "$rendered_unit"' EXIT
sed \
  -e "s|@USER@|$(escape_sed "$USER_NAME")|g" \
  -e "s|@HOME@|$(escape_sed "$USER_HOME")|g" \
  -e "s|@REPO@|$(escape_sed "$REPO")|g" \
  -e "s|@UV@|$(escape_sed "$UV_BIN")|g" \
  "$UNIT" > "$rendered_unit"
install -m 644 "$rendered_unit" "/etc/systemd/system/$SERVICE_NAME.service"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME.service"
if systemctl is-enabled --quiet l6-robot.service 2>/dev/null; then
  systemctl disable l6-robot.service
  echo "disabled legacy l6-robot.service"
fi
# Deliberately not restarted here: a re-run must not take a live robot down
# mid-demo. A changed unit waits until someone asks for it.
echo "enabled (if the unit changed: systemctl restart $SERVICE_NAME)"

say "desktop: disable it to save memory"
# Takes effect at the next boot; it deliberately does not kill your session now.
systemctl set-default multi-user.target
# The over-current popup is a desktop applet, so this retires it too.

say "Wi-Fi: create the robot's network"
wifi_dev="$(nmcli -t -f DEVICE,TYPE device status | awk -F: '$2=="wifi"{print $1; exit}')"
[[ -n "$wifi_dev" ]] || { echo "no wifi device found" >&2; exit 1; }
if ! nmcli -t -f NAME con show | grep -qx "$HOTSPOT_NAME"; then
  nmcli con add type wifi ifname "$wifi_dev" con-name "$HOTSPOT_NAME" ssid "$SSID" >/dev/null
fi
# A 5 GHz non-DFS channel keeps the MJPEG stream responsive in crowded rooms.
# Set the SSID on every run so profile changes reach an existing connection.
nmcli con modify "$HOTSPOT_NAME" \
  802-11-wireless.ssid "$SSID" \
  802-11-wireless.mode ap \
  802-11-wireless.band "$BAND" 802-11-wireless.channel "$CHANNEL" \
  802-11-wireless.powersave 2 \
  ipv4.method shared ipv4.addresses "$AP_IP/24" ipv6.method ignore \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$PSK" \
  connection.autoconnect yes connection.autoconnect-priority 100
# Prevent saved client networks from taking the radio at boot. Use UUIDs
# because connection names may contain colons.
while IFS=: read -r uuid type; do
  [[ "$type" == "802-11-wireless" ]] || continue
  name="$(nmcli -g connection.id con show "$uuid")"
  [[ "$name" != "$HOTSPOT_NAME" ]] || continue
  echo "  parking saved network: $name (autoconnect off)"
  nmcli con modify "$uuid" connection.autoconnect no
done < <(nmcli -t -f UUID,TYPE con show)
# Don't bounce an access point that is already serving: re-running this script
# should not drop the phones currently connected to it.
if nmcli -t -f NAME con show --active | grep -qx "$HOTSPOT_NAME"; then
  # Profile changes need a reconnect, but do not drop current clients here.
  info="$(iw dev "$wifi_dev" info 2>/dev/null || true)"
  live_ch="$(awk '$1=="channel"{print $2}' <<<"$info")"
  live_ssid="$(sed -n 's/^[[:space:]]*ssid //p' <<<"$info")"
  if [[ -n "$live_ch" && "$live_ch" != "$CHANNEL" ]] ||
     [[ -n "$live_ssid" && "$live_ssid" != "$SSID" ]]; then
    echo "hotspot is up as \"$live_ssid\" on channel $live_ch;"
    echo "  the profile now says \"$SSID\" on channel $CHANNEL"
    echo "  to apply it (drops connected clients): nmcli con up $HOTSPOT_NAME"
  else
    echo "hotspot already up"
  fi
else
  nmcli con up "$HOTSPOT_NAME" >/dev/null
fi
echo "hotspot: $SSID / $PSK at $AP_IP"

cat <<EOF

Done. Reboot, then from a phone:
  1. join Wi-Fi "$SSID", password "$PSK"
  2. open https://$AP_IP:8765
  3. accept the certificate once (or install it from https://$AP_IP:8765/cert.crt
     to stop being asked; see docs/phone.md)

Maintenance, from a laptop on the same Wi-Fi:
  ssh $USER_NAME@$AP_IP
  journalctl -u $SERVICE_NAME -f
EOF
