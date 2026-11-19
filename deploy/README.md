# Deploying the Streamlit app

`deploy.sh` runs **on the server** and is idempotent — re-running it is the deploy.
It pulls `main`, installs deps into a venv, writes the systemd unit and nginx
config, and restarts both.

## One-time: create the server

Local Ubuntu VM on macOS:

```bash
brew install --cask multipass
multipass launch 24.04 --name portfolio --cpus 2 --memory 4G --disk 20G
multipass shell portfolio          # you're in
```

## One-time: SSH in with a key

Multipass already injects your key, so this works directly:

```bash
multipass info portfolio           # note the IPv4
ssh ubuntu@<ip>
```

If you want an explicit key (the thing worth practising):

```bash
ssh-keygen -t ed25519 -C portfolio -f ~/.ssh/portfolio      # no passphrase for CI
multipass exec portfolio -- bash -c \
  "echo '$(cat ~/.ssh/portfolio.pub)' >> ~/.ssh/authorized_keys"
ssh -i ~/.ssh/portfolio ubuntu@<ip>
```

## First deploy, by hand

```bash
ssh ubuntu@<ip>
git clone https://github.com/ejung2017/PythonProject.git
./PythonProject/deploy/deploy.sh
```

Then open `http://<ip>/` from the Mac. Useful while debugging:

```bash
systemctl status stockapp
journalctl -u stockapp -f
```

## Automating it

GitHub-hosted runners cannot reach a VM on your laptop, so the runner lives on
the server and polls GitHub outward:

```bash
# On the server. Token from: repo Settings > Actions > Runners > New self-hosted runner
mkdir actions-runner && cd actions-runner
curl -o r.tar.gz -L https://github.com/actions/runner/releases/download/v2.321.0/actions-runner-linux-arm64-2.321.0.tar.gz
tar xzf r.tar.gz
./config.sh --url https://github.com/ejung2017/PythonProject --token <TOKEN>
sudo ./svc.sh install && sudo ./svc.sh start
```

Push to `main` touching `project2/` and `.github/workflows/deploy-app.yml` runs.

## When you move to a real VPS

Swap the two steps in `deploy-app.yml` for an SSH push — the server-side script
does not change:

```yaml
    runs-on: ubuntu-latest
    steps:
      - uses: webfactory/ssh-agent@v0.9.0
        with:
          ssh-private-key: ${{ secrets.DEPLOY_KEY }}
      - run: |
          ssh -o StrictHostKeyChecking=accept-new ubuntu@${{ secrets.DEPLOY_HOST }} \
            'bash -s' < deploy/deploy.sh
```

`DEPLOY_KEY` is the private half of the keypair above, `DEPLOY_HOST` the IP —
both as repo secrets (Settings > Secrets and variables > Actions).

## Notes

- nginx proxies `:80` to Streamlit on `127.0.0.1:8501`. Streamlit uses a
  websocket, so the `Upgrade`/`Connection` headers are required, not optional.
- TA-Lib wraps a C library. `deploy.sh` tries the wheel first and builds from
  source if that fails — the one step likely to need patience on arm64.
