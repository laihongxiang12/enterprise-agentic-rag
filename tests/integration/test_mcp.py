"""Real MCP protocol handshake and read-only tool integration tests."""

import pytest
from mcp import Client, MCPError

from app.mcp.demo import DEMO_DOCUMENT_ID, build_demo_connection
from app.mcp.server import build_mcp_server
from app.schemas.documents import Principal


def principal(*, scopes: tuple[str, ...] = ("tools:get_doc", "tools:search_docs")) -> Principal:
    return Principal(
        user_id="alice",
        tenant_id="tenant-a",
        group_ids=["hr"],
        scopes=list(scopes),
    )


async def test_client_lists_only_two_read_only_tools_and_calls_both() -> None:
    server = build_mcp_server(await build_demo_connection(principal()))

    async with Client(server, raise_exceptions=False) as client:
        listed = await client.list_tools()
        search = await client.call_tool(
            "search_docs",
            {"query": "年假申请提前多久", "limit": 3},
        )
        document = await client.call_tool("get_doc", {"document_id": DEMO_DOCUMENT_ID})

    assert [tool.name for tool in listed.tools] == ["search_docs", "get_doc"]
    schemas = {tool.name: tool.input_schema for tool in listed.tools}
    assert set(schemas["search_docs"]["properties"]) == {"query", "limit"}
    assert set(schemas["get_doc"]["properties"]) == {"document_id"}
    assert search.is_error is False
    assert search.structured_content is not None
    assert search.structured_content["count"] == 1
    assert search.structured_content["items"][0]["document_id"] == DEMO_DOCUMENT_ID
    assert document.is_error is False
    assert document.structured_content is not None
    assert document.structured_content["document_id"] == DEMO_DOCUMENT_ID


async def test_protocol_schema_rejects_tenant_injection_and_unknown_tools() -> None:
    server = build_mcp_server(await build_demo_connection(principal()))

    async with Client(server, raise_exceptions=False) as client:
        with pytest.raises(MCPError, match="Unknown tool arguments"):
            await client.call_tool(
                "search_docs",
                {"query": "policy", "limit": 3, "tenant_id": "tenant-b"},
            )
        dangerous = await client.call_tool("shell", {"command": "whoami"})

    assert dangerous.is_error is True


async def test_missing_tool_scope_fails_without_retrieval() -> None:
    server = build_mcp_server(
        await build_demo_connection(principal(scopes=("tools:get_doc",)))
    )

    async with Client(server, raise_exceptions=False) as client:
        result = await client.call_tool("search_docs", {"query": "年假", "limit": 3})

    assert result.is_error is True
    assert "forbidden" in str(result.content)
