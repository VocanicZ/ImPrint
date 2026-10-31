#!/usr/bin/env bash
# Deploy the committed web UI (HEAD) to https://imprinted.rainbowcreation.net/: the code goes to
# /opt/imprinted on the host (systemd unit imprinted.service, behind its own Cloudflare tunnel),
# then the service restarts and must serve this version. One ssh connection, so one password
# prompt (none with a key). publish.sh runs this last; run it alone to redeploy.
# DEPLOY_HOST overrides the host (default root@vps-test.rainbowcreation.net, reached through
# `cloudflared access ssh`).
set -euo pipefail
cd "$(dirname "$0")/.."
host=${DEPLOY_HOST:-root@vps-test.rainbowcreation.net}
ver=$(git show HEAD:imprinted/__init__.py | sed -n 's/^__version__ = "\(.*\)"/\1/p')

# runs on the host as: bash -c "$remote" deploy <version>, the code's tar on stdin
remote=$(cat <<'EOF'
set -euo pipefail
ver=$1
cd /opt/imprinted
new=$(mktemp -d); trap 'rm -rf "$new"' EXIT
tar -xf - -C "$new"
cmp -s "$new/requirements.txt" requirements.txt || pip=1
rm -rf imprinted web && cp -a "$new/." .                 # results/ (the decode log) stays
[ -z "${pip:-}" ] || .venv/bin/pip install -q -r requirements.txt
chown -R imprinted:imprinted /opt/imprinted
sed -i "s/^Description=.*/Description=Imprinted $ver web UI/" /etc/systemd/system/imprinted.service
systemctl daemon-reload && systemctl restart imprinted
for _ in $(seq 60); do                                   # no curl on the host
  if .venv/bin/python -c 'import sys, urllib.request as u
sys.exit(("v" + sys.argv[1]).encode() not in u.urlopen("http://127.0.0.1:8765/").read())' "$ver" 2>/dev/null; then
    echo "imprinted.rainbowcreation.net serves v$ver"; exit 0
  fi
  sleep 1
done
echo "the web UI did not come back with v$ver: journalctl -u imprinted" >&2; exit 1
EOF
)
git archive HEAD imprinted web web.py requirements.txt pyproject.toml README.md LICENSE |
  ssh -o ProxyCommand="cloudflared access ssh --hostname %h" "$host" "bash -c $(printf '%q' "$remote") deploy $ver"
