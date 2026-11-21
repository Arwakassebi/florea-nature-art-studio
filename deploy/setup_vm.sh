#!/usr/bin/env bash
# À lancer SUR la VM, depuis la racine du dépôt cloné :  bash deploy/setup_vm.sh <nom-DNS-de-la-VM>
# Installe Python, FLOREA Lite, un service systemd et Caddy (HTTPS automatique Let's Encrypt).
set -euo pipefail

DOMAIN="${1:?Donnez le nom DNS de la VM (ex. florea-ab12cd.northeurope.cloudapp.azure.com)}"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_USER="$(id -un)"
REPO_URL="$(git -C "$APP_DIR" remote get-url origin 2>/dev/null || true)"
GH_PATH="$(echo "$REPO_URL" | sed -E 's#^.*github\.com[:/]##; s#\.git$##')"
COLAB_URL="https://colab.research.google.com/github/${GH_PATH:-VOTRE-PSEUDO/florea-nature-art-studio}/blob/main/notebooks/FLOREA_V6.ipynb"

echo ">> 1/5  Swap de 1 Go (la VM n'a que 1 Go de RAM)"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo ">> 2/5  Paquets système et Caddy"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git curl debian-keyring debian-archive-keyring apt-transport-https gpg
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -y && sudo apt-get install -y caddy
fi

echo ">> 3/5  Environnement Python (sans PyTorch : quelques minutes sur une B1s)"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -q --upgrade pip
"$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements-lite.txt"
"$APP_DIR/.venv/bin/pip" install -q --no-deps -e "$APP_DIR"

echo ">> 4/5  Service systemd florea-lite"
sudo tee /etc/systemd/system/florea-lite.service >/dev/null <<UNIT
[Unit]
Description=FLOREA Lite (Gradio)
After=network.target

[Service]
User=$APP_USER
WorkingDirectory=$APP_DIR
Environment=FLOREA_HOST=127.0.0.1
Environment=PORT=7860
Environment=FLOREA_COLAB_URL=$COLAB_URL
Environment=GRADIO_ANALYTICS_ENABLED=False
ExecStart=$APP_DIR/.venv/bin/python -m florea.lite
Restart=on-failure
RestartSec=5
MemoryMax=700M

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now florea-lite

echo ">> 5/5  Caddy : HTTPS automatique pour $DOMAIN"
sudo tee /etc/caddy/Caddyfile >/dev/null <<CADDY
$DOMAIN {
    encode gzip
    reverse_proxy 127.0.0.1:7860
}
CADDY
sudo systemctl restart caddy

echo
echo "✅ Terminé. Attendez ~20 s (entraînement du GCN au démarrage), puis ouvrez : https://$DOMAIN"
echo "   Logs de l'app   : sudo journalctl -u florea-lite -f"
echo "   Logs de Caddy   : sudo journalctl -u caddy -f"
