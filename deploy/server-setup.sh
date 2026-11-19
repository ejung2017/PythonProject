#!/usr/bin/env bash
# Run ONCE on a fresh Ubuntu 24.04 VPS, as root:
#   ssh root@<ip> 'bash -s' < deploy/server-setup.sh
set -euo pipefail

DOMAIN=eunahjung.app
EMAIL=jungeunah98@gmail.com
WEBROOT=/var/www/$DOMAIN

echo "==> packages"
apt-get update -qq
apt-get install -y -qq nginx certbot python3-certbot-nginx rsync ufw

echo "==> deploy user"
# Non-root user for CI to rsync as. Reuses root's authorized key, so the key
# you created the server with keeps working.
id deploy &>/dev/null || adduser --disabled-password --gecos "" deploy
install -d -m 700 -o deploy -g deploy /home/deploy/.ssh
cp /root/.ssh/authorized_keys /home/deploy/.ssh/authorized_keys
chown deploy:deploy /home/deploy/.ssh/authorized_keys
chmod 600 /home/deploy/.ssh/authorized_keys

echo "==> webroot"
install -d -o deploy -g deploy "$WEBROOT"
# Placeholder so nginx has something to serve before the first deploy.
[ -f "$WEBROOT/index.html" ] || echo "<h1>Awaiting first deploy</h1>" > "$WEBROOT/index.html"

echo "==> nginx"
cat > /etc/nginx/sites-available/$DOMAIN <<NGINX
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN www.$DOMAIN;
    root $WEBROOT;
    index index.html;

    # Single-page app: unknown paths fall back to index.html.
    location / {
        try_files \$uri \$uri/ /index.html;
    }

    # Hashed filenames from the Vite build never change contents — cache hard.
    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
NGINX
ln -sf /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/$DOMAIN
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

echo "==> swap"
# 1 GB droplet: swap is the difference between the Streamlit app getting
# OOM-killed (taking nginx down with it) and just running slowly for a moment.
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
fi
grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
# Default swappiness (60) swaps eagerly; 10 keeps swap as a safety net only.
echo 'vm.swappiness=10' > /etc/sysctl.d/99-swappiness.conf
sysctl -q -w vm.swappiness=10
free -h | head -3

echo "==> firewall"
ufw allow OpenSSH >/dev/null
ufw allow 'Nginx Full' >/dev/null
ufw --force enable >/dev/null
ufw status | head -8

echo "==> TLS"
# .app is HSTS-preloaded: without this the site is unreachable, not just insecure.
# Needs the A records live first — rerun this script if it fails here.
certbot --nginx -d "$DOMAIN" -d "www.$DOMAIN" \
  --non-interactive --agree-tos -m "$EMAIL" --redirect

echo "==> done: https://$DOMAIN/"
