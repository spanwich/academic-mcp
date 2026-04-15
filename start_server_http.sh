#!/bin/bash
# start_server_http.sh - Start the HTTP transport MCP server for remote (claude.ai) use.
#
# Unlike start_server.sh, this does NOT auto-start Ollama — this entry point is
# intended for VM deployments that expose only the pre-computed SQLite + ChromaDB
# data, with Ollama-dependent tools disabled via ACADEMIC_MCP_REMOTE_MODE=1.
#
# Required env vars (typically via .env):
#   ACADEMIC_MCP_TOKEN         bearer token clients must present
#   ACADEMIC_MCP_HTTP_PORT     default 8000
#   ACADEMIC_MCP_HTTP_HOST     default 127.0.0.1 (let the reverse proxy handle TLS)
#   ACADEMIC_MCP_REMOTE_MODE   set to 1 on the VM to hide Ollama/Zotero tools

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [ -z "${ACADEMIC_MCP_TOKEN:-}" ]; then
    echo "ACADEMIC_MCP_TOKEN is not set. Refusing to start an unauthenticated HTTP MCP." >&2
    exit 2
fi

# shellcheck disable=SC1091
source venv/bin/activate
exec python -m src.server_http "$@"
