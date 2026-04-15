# academic-mcp on the docker farm

Operational runbook for running academic-mcp as a Docker stack on
`ford@192.168.40.200` (the docker farm), with Ollama on a separate GPU farm
and claude.ai reaching it through a cloudflared named tunnel.

## Architecture at a glance

```
laptop (Zotero)          docker farm (this stack)              GPU farm (Ollama)
 upload_snapshot.sh ───► mirror/zotero/ ──► importer ──┐
 run_import_remote.sh                                  │
                                                       ▼
                         ┌─ mcp (server_http.py) ◄── data/
                         │    ▲
                         │    │ OLLAMA_HOST
 Claude Code CLI ────────┤    ▼
  (LAN HTTP)             │  ollama-tunnel ──── ssh ─────► tmux "ollama serve"
                         │                                iamfo470@deep.otago.ac.nz
 claude.ai ──────────────┤
  (HTTPS)                └─ cloudflared (named tunnel, public URL)
```

Four containers: `mcp` (long-running), `ollama-tunnel` (autossh sidecar),
`cloudflared` (public entry), `importer` (one-shot, profile `import`).

## First-time setup

On the docker farm:

```bash
# 1. Layout
sudo mkdir -p /srv/academic-mcp/{secrets,inbox,mirror,data}
sudo chown -R ford:ford /srv/academic-mcp
cd /srv/academic-mcp

# 2. Put compose.yml next to the data. Cleanest: clone the repo and symlink.
git clone <repo-url> /srv/academic-mcp/repo
ln -s repo/docker/compose.yml compose.yml

# 3. SSH key for the Ollama tunnel. This is the docker-farm-specific key
#    whose pubkey is already in iamfo470@deep.otago.ac.nz:~/.ssh/authorized_keys.
cp ~/path/to/gpu_farm_key secrets/gpu_farm_id_ed25519
chmod 400 secrets/gpu_farm_id_ed25519
touch secrets/tunnel_known_hosts       # autossh writes to this on first run

# 4. Env vars
cp repo/docker/.env.example .env
# edit .env: fill in ACADEMIC_MCP_TOKEN (openssl rand -hex 32) and CF_TUNNEL_TOKEN
```

## Create the cloudflared named tunnel (one-time)

Done once, from any machine where you can log in to Cloudflare Zero Trust
(usually your laptop with a browser). The token is a connector install token
tied to a named tunnel and its configured ingress.

1. `cloudflared tunnel login`
2. `cloudflared tunnel create academic-mcp`
3. `cloudflared tunnel route dns academic-mcp mcp.<your-zone>`
4. In the Cloudflare Zero Trust dashboard → Networks → Tunnels → `academic-mcp`
   → Public hostnames, add:
   - Subdomain: `mcp`, Domain: `<your-zone>`, Service type: `HTTP`, URL: `mcp:8000`.
5. Copy the connector install token from the tunnel's Overview page into
   `CF_TUNNEL_TOKEN` in `/srv/academic-mcp/.env`.

## Build and bring up the stack

```bash
cd /srv/academic-mcp
docker compose build
docker compose up -d mcp ollama-tunnel cloudflared
docker compose ps
```

`mcp` will probably fail its healthcheck on first boot because the tunnel
container hasn't accepted the GPU farm's host key yet. One-shot:

```bash
# Poke the tunnel so autossh TOFUs the host key into secrets/tunnel_known_hosts
docker compose logs ollama-tunnel
docker compose restart mcp
```

After that, both stay healthy unless Ollama is down.

## Start Ollama on the GPU farm

**Manual, per the deployment's design.** There is no auto-restart. Do this
once after every GPU farm reboot:

```bash
ssh iamfo470@deep.otago.ac.nz
tmux new -s ollama
OLLAMA_HOST=127.0.0.1:11434 ollama serve
# Ctrl-b d to detach
```

Verify the required models are pulled:

```bash
ssh iamfo470@deep.otago.ac.nz tmux send-keys -t ollama 'ollama list' Enter
# Should include: qwen3:8b, nomic-embed-text
# If not: ssh in, `tmux attach -t ollama`, Ctrl-c, `ollama pull <model>`, restart serve
```

If Ollama dies, MCP tool calls that hit it (`search_content`, `query_paper`)
and all importer runs will fail with a clear connection error. Fix by
restarting the tmux session; no docker-farm-side action needed.

## Importing papers (laptop-driven, two steps)

```bash
# 1. On the laptop: snapshot Zotero and push it to the docker farm.
./scripts/upload_snapshot.sh

# 2. On the laptop: run the importer one-shot on the docker farm.
./scripts/run_import_remote.sh --collection "seL4 verification"
#   or --all, or --item <key>, etc.

# 3. Restart mcp so it reopens papers.db + chroma with the new data.
ssh ford@192.168.40.200 'cd /srv/academic-mcp && docker compose restart mcp'
```

Why three steps instead of one: the user preference is explicit manual
control between upload / extract / serve. No hidden chaining.

## Client config — Claude Code on the laptop (LAN)

Add to `~/.config/claude-code/mcp.json`:

```json
{
  "mcpServers": {
    "academic-papers": {
      "transport": "http",
      "url": "http://192.168.40.200:8000/mcp",
      "headers": {
        "Authorization": "Bearer <ACADEMIC_MCP_TOKEN from .env>"
      }
    }
  }
}
```

Quick sanity check from the laptop without Claude Code:

```bash
TOKEN=<paste>
curl -sf -H "Authorization: Bearer $TOKEN" http://192.168.40.200:8000/healthz
# → {"status":"ok"}
```

## Client config — claude.ai (HTTPS via cloudflared)

In claude.ai → Settings → Connectors → Add custom connector:

- URL: `https://mcp.<your-zone>/mcp`
- Auth: Bearer token = `ACADEMIC_MCP_TOKEN` (same token as Claude Code)

Verify from any internet-connected shell:

```bash
curl -sf -H "Authorization: Bearer $TOKEN" https://mcp.<your-zone>/healthz
# → {"status":"ok"}
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `curl` to `/healthz` returns 401 | Missing / wrong bearer token | Check `.env` and client config match |
| `search_content` or `query_paper` hangs | Ollama not running on GPU farm | `ssh iamfo470@deep.otago.ac.nz`, `tmux attach -t ollama`, restart `ollama serve` |
| `docker compose logs ollama-tunnel` shows `Permission denied (publickey)` | Wrong SSH key mounted, or key not in GPU farm `authorized_keys` | Re-check `secrets/gpu_farm_id_ed25519` and rerun `ssh-copy-id` on the GPU farm |
| `docker compose logs ollama-tunnel` shows `Host key verification failed` | First-run TOFU didn't stick | Delete `secrets/tunnel_known_hosts`, `touch` it, restart tunnel; the `accept-new` policy will re-accept |
| Importer runs but MCP doesn't see new papers | `mcp` container is holding stale `papers.db` handles | `docker compose restart mcp` |
| cloudflared container restarts repeatedly | `CF_TUNNEL_TOKEN` invalid or tunnel deleted in dashboard | Regenerate token in the Zero Trust dashboard, update `.env`, `docker compose up -d cloudflared` |
| MCP 500 on tool calls, logs show `OperationalError: unable to open database` | `/data` bind mount missing or wrong perms | Check `/srv/academic-mcp/data` exists and is owned by the container UID |

## Stopping and cleanup

```bash
docker compose down                         # stop stack, keep volumes
docker compose down --volumes                # also drop compose-managed volumes (none by default)
sudo rm -rf /srv/academic-mcp/data           # nuke papers.db + chroma (requires re-import)
sudo rm -rf /srv/academic-mcp/mirror/zotero  # nuke Zotero mirror
```

Backups of `/srv/academic-mcp/data/` are out of scope for this runbook but
worth doing — that directory is the product of hours of LLM extraction and
is the only thing that can't be rebuilt quickly from the laptop.
