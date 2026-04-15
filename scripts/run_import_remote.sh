#!/bin/bash
# run_import_remote.sh — trigger a one-shot importer container on the docker farm.
#
# Laptop-side. Thin SSH wrapper that passes all arguments through to
# `zotero_import.py` inside a fresh container (via compose profile "import").
# Requires a prior successful run of `upload_snapshot.sh` so the docker farm
# has a fresh mirror under /srv/academic-mcp/mirror/zotero/.
#
# Example:
#   ./scripts/run_import_remote.sh --collection "seL4 verification"
#   ./scripts/run_import_remote.sh --all
#   ./scripts/run_import_remote.sh --item klein_2009_sel4
#
# Env overrides:
#   REMOTE       default: ford@192.168.40.200
#   REMOTE_ROOT  default: /srv/academic-mcp

set -euo pipefail

REMOTE="${REMOTE:-ford@192.168.40.200}"
REMOTE_ROOT="${REMOTE_ROOT:-/srv/academic-mcp}"

if [ "$#" -eq 0 ]; then
    cat >&2 <<EOF
usage: $0 <zotero_import.py args>

examples:
  $0 --collection "seL4 verification"
  $0 --all
  $0 --item klein_2009_sel4
  $0 --list-collections
EOF
    exit 2
fi

# -t so the container gets a TTY and zotero_import.py's progress output
# streams back to the laptop terminal live.
ssh -t "$REMOTE" "cd ${REMOTE_ROOT} && docker compose --profile import run --rm importer $*"

echo
echo "==> Import finished. Restart mcp to reopen papers.db + chroma:"
echo "    ssh ${REMOTE} 'cd ${REMOTE_ROOT} && docker compose restart mcp'"
