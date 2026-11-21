#!/usr/bin/env bash
# Met à jour l'application sur la VM après un `git push` :  bash deploy/update.sh
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
git -C "$APP_DIR" pull --ff-only
"$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements-lite.txt"
"$APP_DIR/.venv/bin/pip" install -q --no-deps -e "$APP_DIR"
sudo systemctl restart florea-lite
echo "✅ Mis à jour. Statut :"; systemctl --no-pager --lines=3 status florea-lite | head -5
