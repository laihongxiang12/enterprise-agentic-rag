"""MCP 2.3 stdio server with identity fixed by the launching process."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

from mcp import MCPError
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from app.mcp.service import (
    GetDocResult,
    MCPReadService,
    MCPServiceError,
    SearchDocsResult,
)
from app.schemas.documents import Principal


@dataclass(frozen=True, slots=True)
class MCPConnectionContext:
    """Trusted context fixed before stdio starts and invisible to tool schemas."""

    principal: Principal
    workspace_id: str
    service: MCPReadService


class StrictToolArguments:
    """Reject unknown tool arguments before SDK function binding can ignore them."""

    _allowed: ClassVar[dict[str, frozenset[str]]] = {
        "search_docs": frozenset({"query", "limit"}),
        "get_doc": frozenset({"document_id"}),
    }

    async def __call__(
        self,
        ctx: ServerRequestContext[Any, Any],
        call_next: CallNext,
    ) -> HandlerResult:
        if ctx.method == "tools/call" and ctx.params is not None:
            name = ctx.params.get("name")
            arguments = ctx.params.get("arguments", {})
            allowed = self._allowed.get(name) if isinstance(name, str) else None
            if (
                allowed is not None
                and isinstance(arguments, dict)
                and not set(arguments).issubset(allowed)
            ):
                raise MCPError(-32602, "Unknown tool arguments are forbidden")
        return await call_next(ctx)


def build_mcp_server(connection: MCPConnectionContext) -> MCPServer[MCPConnectionContext]:
    """Build a two-tool read-only server for one trusted stdio connection."""

    @asynccontextmanager
    async def lifespan(
        _server: MCPServer[MCPConnectionContext],
    ) -> AsyncIterator[MCPConnectionContext]:
        yield connection

    server: MCPServer[MCPConnectionContext] = MCPServer(
        name="enterprise-agentic-rag",
        version="0.1.0",
        instructions=(
            "Read-only enterprise document retrieval. Identity and workspace are server-owned."
        ),
        lifespan=lifespan,
        middleware=[StrictToolArguments()],
    )
    read_only = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )

    @server.tool(annotations=read_only, structured_output=True)
    async def search_docs(
        query: Annotated[str, Field(min_length=1, max_length=2_000)],
        ctx: Context[MCPConnectionContext],
        limit: Annotated[int, Field(ge=1, le=20)] = 8,
    ) -> SearchDocsResult:
        """Search only documents authorized for the verified connection identity."""
        runtime = ctx.request_context.lifespan_context
        try:
            return await runtime.service.search_docs(
                runtime.principal,
                runtime.workspace_id,
                query,
                limit,
            )
        except MCPServiceError as exc:
            raise ToolError(exc.code) from exc

    @server.tool(annotations=read_only, structured_output=True)
    async def get_doc(
        document_id: Annotated[str, Field(min_length=1, max_length=256)],
        ctx: Context[MCPConnectionContext],
    ) -> GetDocResult:
        """Return metadata only when the document is authorized for this connection."""
        runtime = ctx.request_context.lifespan_context
        try:
            return await runtime.service.get_doc(
                runtime.principal,
                runtime.workspace_id,
                document_id,
            )
        except MCPServiceError as exc:
            raise ToolError(exc.code) from exc

    return server


def main() -> None:
    """Run the explicit local Fake stdio profile; it never loads LLM credentials."""
    from app.mcp.demo import build_demo_connection_from_environment

    connection = asyncio.run(build_demo_connection_from_environment())
    build_mcp_server(connection).run("stdio")


if __name__ == "__main__":
    main()
