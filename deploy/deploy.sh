#!/usr/bin/env bash
# Deploys the Streamlit app. Run ON the server. Idempotent — safe to re-run.
set -euo pipefail

REPO=https://github.com/ejung2017/PythonProject.git
DIR=$HOME/PythonProject
APP=$DIR/project2/stock_prediction
SVC=stockapp

echo "==> source"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch --quiet origin main && git -C "$DIR" reset --hard origin/main
else
  git clone --quiet "$REPO" "$DIR"
fi

echo "==> system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev build-essential nginx

echo "==> python deps"
[ -d "$DIR/.venv" ] || python3 -m venv "$DIR/.venv"
"$DIR/.venv/bin/pip" install --quiet --upgrade pip
# TA-Lib wraps a C library. Try the wheel; fall back to building it from source.
if ! "$DIR/.venv/bin/pip" install --quiet -r "$APP/requirements.txt"; then
  echo "    pip failed — building TA-Lib C library from source"
  if [ ! -f /usr/local/lib/libta-lib.so ]; then
    tmp=$(mktemp -d)
    curl -sSL https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
      | tar xz -C "$tmp" --strip-components=1
    ( cd "$tmp" && ./configure --prefix=/usr/local >/dev/null && make -s && sudo make install >/dev/null )
    sudo ldconfig
  fi
  "$DIR/.venv/bin/pip" install -r "$APP/requirements.txt"
fi

echo "==> systemd service"
sudo tee /etc/systemd/system/$SVC.service >/dev/null <<UNIT
[Unit]
Description=Stock Prediction Streamlit app
After=network.target

[Service]
User=$USER
WorkingDirectory=$APP
ExecStart=$DIR/.venv/bin/streamlit run app.py \
  --server.port 8501 --server.address 127.0.0.1 --server.headless true
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT

echo "==> nginx reverse proxy"
sudo tee /etc/nginx/sites-available/$SVC >/dev/null <<'NGINX'
server {
    listen 80 default_server;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        # Streamlit talks over a websocket — it breaks without these two.
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }
}
NGINX
sudo ln -sf /etc/nginx/sites-available/$SVC /etc/nginx/sites-enabled/$SVC
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t

echo "==> restart"
sudo systemctl daemon-reload
sudo systemctl enable --now $SVC
sudo systemctl restart $SVC nginx

sleep 3
systemctl is-active --quiet $SVC && echo "==> $SVC is up on http://$(hostname -I | awk '{print $1}')/" || {
  echo "==> $SVC FAILED — last 30 log lines:"; journalctl -u $SVC -n 30 --no-pager; exit 1; }
