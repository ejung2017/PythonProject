#!/usr/bin/env bash
# Deploys the Streamlit app behind nginx at https://eunahjung.app/stock/
# Run as root on the server. Idempotent — re-running it IS the deploy.
#   ssh root@<ip> 'bash -s' < deploy/deploy.sh
set -euo pipefail

REPO=https://github.com/ejung2017/PythonProject.git
APPDIR=/srv/stockapp
APP=$APPDIR/project2/stock_prediction
VENV=$APPDIR/.venv
SVC=stockapp
RUNAS=deploy
VHOST=/etc/nginx/sites-available/eunahjung.app
SNIPPET=/etc/nginx/snippets/stock.conf

echo "==> build deps"
apt-get update -qq
apt-get install -y -qq python3-venv python3-dev build-essential pkg-config curl

echo "==> source"
if [ -d "$APPDIR/.git" ]; then
  git -C "$APPDIR" fetch -q origin main
  git -C "$APPDIR" reset -q --hard origin/main
else
  git clone -q "$REPO" "$APPDIR"
fi
chown -R $RUNAS:$RUNAS "$APPDIR"

echo "==> python deps"
sudo -u $RUNAS test -d "$VENV" || sudo -u $RUNAS python3 -m venv "$VENV"
sudo -u $RUNAS "$VENV/bin/pip" install -q --upgrade pip
# TA-Lib wraps a C library. Try the wheel first; build the C lib only if needed.
if ! sudo -u $RUNAS "$VENV/bin/pip" install -q -r "$APP/requirements.txt" 2>/dev/null; then
  echo "    wheel install failed — building TA-Lib C library (slow, one time)"
  if [ ! -f /usr/local/lib/libta-lib.so ]; then
    tmp=$(mktemp -d)
    curl -sSL https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
      | tar xz -C "$tmp" --strip-components=1
    ( cd "$tmp" && ./configure --prefix=/usr/local >/dev/null && make -s && make install >/dev/null )
    ldconfig
    rm -rf "$tmp"
  fi
  sudo -u $RUNAS "$VENV/bin/pip" install -r "$APP/requirements.txt"
fi

echo "==> systemd service"
# Runs as the unprivileged deploy user, bound to localhost only — nginx is the
# only way in. baseUrlPath makes Streamlit emit /stock/-prefixed asset URLs.
cat > /etc/systemd/system/$SVC.service <<UNIT
[Unit]
Description=Stock Prediction Streamlit app
After=network.target

[Service]
User=$RUNAS
WorkingDirectory=$APP
ExecStart=$VENV/bin/streamlit run app.py \
  --server.port 8501 \
  --server.address 127.0.0.1 \
  --server.headless true \
  --server.baseUrlPath stock \
  --browser.gatherUsageStats false
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT

echo "==> nginx location"
mkdir -p "$(dirname $SNIPPET)"
cat > "$SNIPPET" <<'NGINX'
# Streamlit at /stock/ — proxied, not served from disk.
location = /stock { return 301 /stock/; }
location /stock/ {
    proxy_pass http://127.0.0.1:8501/stock/;
    proxy_http_version 1.1;
    # Streamlit's UI runs over a websocket; it silently fails without these.
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 86400;
}
NGINX
# Insert the include into the TLS server block, once. "index index.html;"
# appears only in that block, so it is a safe anchor.
if ! grep -q "snippets/stock.conf" "$VHOST"; then
  sed -i 's|^\(\s*\)index index.html;|\1index index.html;\n\1include /etc/nginx/snippets/stock.conf;|' "$VHOST"
fi
nginx -t

echo "==> restart"
systemctl daemon-reload
systemctl enable -q --now $SVC
systemctl restart $SVC
systemctl reload nginx

sleep 5
if systemctl is-active --quiet $SVC; then
  echo "==> $SVC active — https://eunahjung.app/stock/"
else
  echo "==> $SVC FAILED:"; journalctl -u $SVC -n 30 --no-pager; exit 1
fi
