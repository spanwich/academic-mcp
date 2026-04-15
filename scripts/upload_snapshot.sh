#!/bin/bash
# upload_snapshot.sh — push a full Zotero snapshot to the docker farm.
#
# Laptop-side. Packs ~/Zotero into a timestamped tarball, scp's it to the
# docker farm's inbox/, then SSHes in and atomically swaps it into
# mirror/zotero/ so the `mcp` container always sees either the old snapshot
# or the new one — never a half-written mirror.
#
# After this succeeds, run:
#   ./scripts/run_import_remote.sh --collection "<name>"   # or --all
# and then (once import finishes):
#   ssh $REMOTE 'cd /srv/academic-mcp && docker compose restart mcp'
#
# Env overrides:
#   REMOTE       default: ford@192.168.40.200
#   REMOTE_ROOT  default: /srv/academic-mcp
#   ZOTERO_DIR   default: $HOME/Zotero

set -euo pipefail

REMOTE="${REMOTE:-ford@192.168.40.200}"
REMOTE_ROOT="${REMOTE_ROOT:-/srv/academic-mcp}"
ZOTERO_DIR="${ZOTERO_DIR:-$HOME/Zotero}"

if [ ! -d "$ZOTERO_DIR" ]; then
    echo "error: ZOTERO_DIR '$ZOTERO_DIR' does not exist" >&2
    exit 1
fi
if [ ! -f "$ZOTERO_DIR/zotero.sqlite" ]; then
    echo "error: '$ZOTERO_DIR/zotero.sqlite' not found — is Zotero installed here?" >&2
    exit 1
fi

STAMP=$(date +%Y%m%d-%H%M%S)
TAR_NAME="zotero-snapshot-${STAMP}.tar.gz"
TAR_LOCAL="/tmp/${TAR_NAME}"

ZOTERO_PARENT="$(dirname "$ZOTERO_DIR")"
ZOTERO_BASENAME="$(basename "$ZOTERO_DIR")"

echo "==> Packing $ZOTERO_DIR -> $TAR_LOCAL"
tar czf "$TAR_LOCAL" -C "$ZOTERO_PARENT" "$ZOTERO_BASENAME"
SIZE=$(du -h "$TAR_LOCAL" | cut -f1)
echo "    $SIZE"

echo "==> Uploading to ${REMOTE}:${REMOTE_ROOT}/inbox/"
ssh "$REMOTE" "mkdir -p ${REMOTE_ROOT}/inbox ${REMOTE_ROOT}/mirror"
scp "$TAR_LOCAL" "${REMOTE}:${REMOTE_ROOT}/inbox/${TAR_NAME}"

echo "==> Swapping mirror atomically on docker farm"
ssh "$REMOTE" ZOTERO_BASENAME="$ZOTERO_BASENAME" TAR_NAME="$TAR_NAME" REMOTE_ROOT="$REMOTE_ROOT" bash -s <<'REMOTE_EOF'
set -euo pipefail
cd "$REMOTE_ROOT"

rm -rf mirror/.zotero.new
mkdir -p mirror/.zotero.new

# --strip-components=1 drops the leading "Zotero/" dir from the tarball so
# the mirror root directly contains zotero.sqlite, storage/, etc.
tar xzf "inbox/${TAR_NAME}" -C mirror/.zotero.new --strip-components=1

rm -rf mirror/.zotero.old 2>/dev/null || true
if [ -d mirror/zotero ]; then
    mv mirror/zotero mirror/.zotero.old
fi
mv mirror/.zotero.new mirror/zotero
rm -rf mirror/.zotero.old

# Keep the last 3 snapshot tarballs, prune older ones.
ls -1t inbox/zotero-snapshot-*.tar.gz 2>/dev/null | tail -n +4 | xargs -r rm --
REMOTE_EOF

rm -f "$TAR_LOCAL"
echo "==> Done. Remote mirror is now ${REMOTE_ROOT}/mirror/zotero/"
echo
echo "Next steps:"
echo "  ./scripts/run_import_remote.sh --collection \"<name>\"   # or --all"
echo "  ssh ${REMOTE} 'cd ${REMOTE_ROOT} && docker compose restart mcp'"
