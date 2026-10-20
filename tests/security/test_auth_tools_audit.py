"""Authentication, tool registry, and audit-data security regressions."""

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.core.dependencies import VerifiedIdentity
from app.core.security import (
    AuthenticationError,
    BasicPolicyEngine,
    DevTokenService,
    audit_safe_details,
)
from app.schemas.documents import Principal
from app.tools.policy import DEFAULT_TOOL_POLICY, ToolRegistration

SECRET = "stage-17-local-secret-that-is-at-least-32-characters"


def settings() -> Settings:
    return Settings(
        app_env="local",
        app_mode="fake",
        bind_host="127.0.0.1",
        dev_auth_enabled=True,
        dev_auth_secret=SecretStr(SECRET),
    )


def test_forged_token_and_oversized_header_fail_closed() -> None:
    service = DevTokenService(settings())
    forged = jwt.encode(
        {
            "sub": "mallory",
            "tenant_id": "tenant-a",
            "iss": settings().dev_auth_issuer,
            "aud": settings().dev_auth_audience,
            "iat": datetime.now(UTC),
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        "wrong-secret-that-is-also-long-enough-123456",
        algorithm="HS256",
    )
    with pytest.raises(AuthenticationError):
        service.verify(forged)
    with pytest.raises(AuthenticationError):
        service.verify_bearer(f"Bearer {'x' * 8_200}")


@pytest.mark.parametrize(
    "claim_update",
    [
        {"sub": "alice\nadmin"},
        {"tenant_id": "tenant-a\x00tenant-b"},
        {"groups": ["g"] * 129},
        {"scopes": ["x" * 257]},
    ],
)
def test_signed_but_malformed_identity_claims_are_rejected(
    claim_update: dict[str, object],
) -> None:
    configured = settings()
    now = datetime.now(UTC)
    claims: dict[str, object] = {
        "sub": "alice",
        "tenant_id": "tenant-a",
        "groups": ["hr"],
        "scopes": ["query:read"],
        "iss": configured.dev_auth_issuer,
        "aud": configured.dev_auth_audience,
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }
    claims.update(claim_update)
    token = jwt.encode(claims, SECRET, algorithm="HS256")
    with pytest.raises(AuthenticationError):
        DevTokenService(configured).verify(token)


def test_default_registry_contains_only_read_only_document_search() -> None:
    assert DEFAULT_TOOL_POLICY.registered_names == ("get_doc", "search_docs")
    principal = Principal(
        user_id="alice",
        tenant_id="tenant-a",
        scopes=[
            "tools:get_doc",
            "tools:search_docs",
            "tools:web",
            "tools:sql",
            "tools:shell",
            "tools:python",
        ],
    )
    policy = BasicPolicyEngine()
    assert policy.authorize_tool(principal, "get_doc")
    assert policy.authorize_tool(principal, "search_docs")
    for dangerous in ("web", "sql", "shell", "python", "filesystem_write"):
        assert not policy.authorize_tool(principal, dangerous)
    assert not policy.authorize_tool(principal, "search_docs\n")
    assert not policy.authorize_tool(principal, "SEARCH_DOCS")


def test_dangerous_registration_is_rejected_at_startup() -> None:
    with pytest.raises(ValueError, match="dangerous tools"):
        ToolRegistration(name="shell", required_scope="tools:shell")


def test_audit_details_drop_prompts_content_tokens_and_keys() -> None:
    details = {
        "query": "show tenant-b payroll",
        "document_content": "ignore instructions",
        "authorization": "Bearer secret-token",
        "api_key": "sk-not-a-real-key",
        "result": "denied",
    }
    assert audit_safe_details(details) == {}


def test_normal_token_round_trip_still_works() -> None:
    service = DevTokenService(settings())
    token = service.issue(
        VerifiedIdentity(
            user_id="alice",
            tenant_id="tenant-a",
            group_ids=("hr",),
            scopes=("query:read",),
        )
    )
    assert service.verify(token).tenant_id == "tenant-a"
