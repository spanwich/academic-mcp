"""
HTTP transport entry point for the Academic Paper MCP server.

Wraps the existing low-level `Server` instance from `src.server` in a
Starlette ASGI app speaking MCP Streamable HTTP, gated by a bearer token.

When `ACADEMIC_MCP_REMOTE_MODE=1`, tools that require Ollama or live Zotero
access are hidden from `list_tools` and rejected in `call_tool`. This lets
the same code run on a VM that ships only `data/papers.db` + `data/chroma/`
without Ollama or Zotero.

Launch: `python -m src.server_http`  (or via `start_server_http.sh`).
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import AsyncIterator

import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import TextContent
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

from src.config import get_config
from src.server import TOOLS, call_tool as _orig_call_tool, server

logger = logging.getLogger("academic_mcp.http")


REMOTE_BLOCKED_TOOLS: frozenset[str] = frozenset(
    {
        "zotero_list_collections",  # needs live Zotero
        "zotero_list_items",        # needs live Zotero
        "search_content",           # needs Ollama for query embeddings
        "query_paper",              # needs Ollama LLM
    }
)


def _install_remote_allowlist() -> None:
    """Re-register list_tools/call_tool on the shared `server` to filter blocked tools.

    Safe because `Server.list_tools()` / `Server.call_tool()` store handlers in
    `server.request_handlers` keyed by request type, so a second registration
    simply replaces the first.
    """
    filtered_tools = [t for t in TOOLS if t.name not in REMOTE_BLOCKED_TOOLS]
    logger.info(
        "Remote mode: exposing %d/%d tools, blocked=%s",
        len(filtered_tools),
        len(TOOLS),
        sorted(REMOTE_BLOCKED_TOOLS),
    )

    @server.list_tools()
    async def list_tools_remote():
        return filtered_tools

    @server.call_tool()
    async def call_tool_remote(name: str, arguments: dict):
        if name in REMOTE_BLOCKED_TOOLS:
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Tool '{name}' is disabled in this deployment "
                        "(no Ollama/Zotero available)."
                    ),
                )
            ]
        return await _orig_call_tool(name, arguments)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Authorization header doesn't match the configured token.

    Exempts the health endpoint so load balancers / tunnels can probe without a token.
    """

    def __init__(self, app, token: str):
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/healthz":
            return await call_next(request)

        header = request.headers.get("authorization", "")
        expected = f"Bearer {self._token}"
        if not self._token or header != expected:
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="academic-mcp"'},
            )
        return await call_next(request)


async def healthz(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def build_app() -> Starlette:
    config = get_config()

    if config.mcp_remote_mode:
        _install_remote_allowlist()

    if not config.mcp_token:
        logger.error(
            "ACADEMIC_MCP_TOKEN is not set. Refusing to start an unauthenticated HTTP MCP."
        )
        raise SystemExit(2)

    session_manager = StreamableHTTPSessionManager(
        app=server,
        stateless=True,
    )

    async def handle_mcp(scope: Scope, receive: Receive, send: Send) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            logger.info(
                "academic-mcp HTTP listening on http://%s:%d/mcp (remote_mode=%s)",
                config.mcp_http_host,
                config.mcp_http_port,
                config.mcp_remote_mode,
            )
            yield

    return Starlette(
        debug=False,
        lifespan=lifespan,
        middleware=[Middleware(BearerAuthMiddleware, token=config.mcp_token)],
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Mount("/mcp", app=handle_mcp),
        ],
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    config = get_config()
    app = build_app()
    uvicorn.run(
        app,
        host=config.mcp_http_host,
        port=config.mcp_http_port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
