#!/bin/bash
# start_server.sh - Start Ollama and MCP server together

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=== Academic Paper MCP Server v3 ==="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Ollama not found. Please install it first:${NC}"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

# Check if Ollama is already running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama already running${NC}"
else
    echo "Starting Ollama..."
    nohup ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ollama started (PID: $OLLAMA_PID)${NC}"
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Failed to start Ollama"
        exit 1
    fi
fi

# Check if model is available
MODEL="${ACADEMIC_LLM_MODEL:-qwen2.5:3b}"
if ! ollama list | grep -q "${MODEL%%:*}"; then
    echo -e "${YELLOW}Model $MODEL not found. Pulling...${NC}"
    ollama pull "$MODEL"
fi

# Activate venv and start MCP server
echo "Starting MCP server..."
source venv/bin/activate
exec python -m src.server "$@"
