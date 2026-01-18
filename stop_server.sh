#!/bin/bash
# stop_server.sh - Gracefully stop the MCP server
#
# Sends SIGTERM to the server process for graceful shutdown.

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Find the MCP server process
MCP_PID=$(pgrep -f "python -m src.server" | head -1)

if [ -z "$MCP_PID" ]; then
    echo -e "${YELLOW}MCP server is not running.${NC}"
    exit 0
fi

echo "Found MCP server (PID: $MCP_PID)"
echo "Sending SIGTERM for graceful shutdown..."

# Send SIGTERM (graceful)
kill -TERM "$MCP_PID" 2>/dev/null

# Wait for process to exit (up to 10 seconds)
for i in {1..10}; do
    if ! kill -0 "$MCP_PID" 2>/dev/null; then
        echo -e "${GREEN}✓ MCP server stopped gracefully${NC}"
        exit 0
    fi
    sleep 1
done

# Still running? Send SIGKILL
echo -e "${YELLOW}Server didn't stop gracefully, sending SIGKILL...${NC}"
kill -KILL "$MCP_PID" 2>/dev/null

if ! kill -0 "$MCP_PID" 2>/dev/null; then
    echo -e "${GREEN}✓ MCP server stopped${NC}"
else
    echo -e "${RED}✗ Failed to stop MCP server${NC}"
    exit 1
fi
