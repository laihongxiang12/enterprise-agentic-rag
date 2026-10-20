"""Minimal local client that performs a real MCP stdio handshake."""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters
from pydantic import SecretStr

from app.core.config import Settings
from app.core.dependencies import VerifiedIdentity
from app.core.security import DevTokenService


async def run() -> dict[str, object]:
    ephemeral_secret = secrets.token_urlsafe(32)
    auth_settings = Settings(
        app_env="local",
        app_mode="fake",
        bind_host="127.0.0.1",
        dev_auth_enabled=True,
        dev_auth_secret=SecretStr(ephemeral_secret),
        _env_file=None,
    )
    auth = DevTokenService(auth_settings)
    token = auth.issue(
        VerifiedIdentity(
            user_id="mcp-demo-user",
            tenant_id="northstar-demo",
            group_ids=("hr",),
            scopes=("tools:get_doc", "tools:search_docs"),
        )
    )
    env = {
        "DEV_AUTH_SECRET": ephemeral_secret,
        "DEV_AUTH_ISSUER": auth_settings.dev_auth_issuer,
        "DEV_AUTH_AUDIENCE": auth_settings.dev_auth_audience,
        "MCP_DEV_BEARER_TOKEN": token,
        "MCP_WORKSPACE_ID": "policies",
    }
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp.server"],
        env=env,
        cwd=Path.cwd(),
    )
    async with Client(parameters, read_timeout_seconds=10) as client:
        tools = await client.list_tools()
        search = await client.call_tool(
            "search_docs",
            {"query": "年假申请提前多久", "limit": 3},
        )
        document = await client.call_tool(
            "get_doc",
            {"document_id": "demo-annual-leave"},
        )
    return {
        "tools": sorted(tool.name for tool in tools.tools),
        "search": search.model_dump(mode="json", by_alias=True),
        "document": document.model_dump(mode="json", by_alias=True),
    }


def main() -> None:
    print(json.dumps(asyncio.run(run()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
