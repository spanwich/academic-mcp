# ollama-tunnel sidecar image.
# Single purpose: autossh -N -L to the GPU farm's Ollama, nothing else.
# Kept separate from the main app image so it stays tiny and has a minimal
# attack surface (only SSH client tooling).

FROM alpine:3.20

RUN apk add --no-cache openssh-client autossh

# SSH key is bind-mounted at runtime to /root/.ssh/id_ed25519 (read-only).
# Make sure the home dir exists with correct perms.
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# No default command — compose.yml specifies the exact autossh invocation,
# including the remote host, so the image itself stays credential-free.
ENTRYPOINT ["/usr/bin/autossh"]
