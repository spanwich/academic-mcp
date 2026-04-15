# academic-mcp main image.
# Shared by the `mcp` long-running service and the `importer` one-shot.
# Build context is the repo root (compose.yml uses `context: ..`).

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false

# System deps for PyMuPDF (no special libs at runtime, but build-essential is
# occasionally needed for transitive wheels on slim). curl kept for healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so the layer caches well across source edits.
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy application source. Everything below this line invalidates on source edits.
COPY src/ /app/src/
COPY zotero_import.py /app/zotero_import.py
COPY start_server_http.sh /app/start_server_http.sh
RUN chmod +x /app/start_server_http.sh

# Directories the containers will mount over — declared so fresh containers
# have them even if bind mounts are missing (e.g. during a smoke test).
RUN mkdir -p /mirror /data

# Default to the HTTP server; the importer service overrides `command` in compose.
CMD ["./start_server_http.sh"]
