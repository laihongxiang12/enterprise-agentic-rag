"""Signed development identity and fail-closed ACL policy tests."""

import jwt
import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.core.dependencies import VerifiedIdentity
from app.core.security import AuthenticationError, BasicPolicyEngine, DevTokenService
from app.schemas.documents import DocumentMeta, Principal
from app.stores.fakes import MemoryAuthorizationMetadataStore

STRONG_SECRET = "local-test-secret-that-is-at-least-32-characters"


def dev_settings(**updates: object) -> Settings:
    values: dict[str, object] = {
        "app_env": "local",
        "app_mode": "fake",
        "bind_host": "127.0.0.1",
        "dev_auth_enabled": True,
        "dev_auth_secret": SecretStr(STRONG_SECRET),
    }
    values.update(updates)
    return Settings.model_validate(values)


def document(
    *,
    tenant_id: str,
    document_id: str,
    groups: list[str] | None = None,
    visibility: str = "restricted",
    status: str = "ready",
    acl_version: int = 1,
) -> DocumentMeta:
    return DocumentMeta.model_validate(
        {
            "tenant_id": tenant_id,
            "workspace_id": "hr",
            "document_id": document_id,
            "version": 1,
            "title": f"Policy {document_id}",
            "source_uri": f"fixture:///{document_id}.md",
            "source_type": "markdown",
            "visibility": visibility,
            "acl_groups": groups or [],
            "content_hash": "a" * 64,
            "acl_version": acl_version,
            "index_version": f"index-{document_id}-a{acl_version}",
            "status": status,
        }
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"app_env": "production"},
        {"app_mode": "real", "llm_api_key": SecretStr("configured")},
        {"bind_host": "0.0.0.0"},
        {"dev_auth_secret": SecretStr("too-short")},
        {"dev_auth_secret": None},
    ],
)
def test_dev_auth_rejects_nonlocal_or_weak_configuration(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        dev_settings(**updates)


def test_signed_dev_token_round_trip_builds_principal() -> None:
    service = DevTokenService(dev_settings())
    token = service.issue(
        VerifiedIdentity(
            user_id="alice",
            tenant_id="tenant-a",
            group_ids=("hr", "employees"),
            scopes=("documents:read",),
        )
    )
    principal = service.verify_bearer(f"Bearer {token}")
    assert principal == Principal(
        user_id="alice",
        tenant_id="tenant-a",
        group_ids=["employees", "hr"],
        scopes=["documents:read"],
    )


def test_wrong_secret_and_bad_bearer_are_rejected() -> None:
    identity = VerifiedIdentity(user_id="alice", tenant_id="tenant-a")
    first = DevTokenService(dev_settings())
    second = DevTokenService(
        dev_settings(dev_auth_secret=SecretStr("another-secret-that-is-long-enough-1234"))
    )
    token = first.issue(identity)
    with pytest.raises(AuthenticationError):
        second.verify(token)
    with pytest.raises(AuthenticationError):
        first.verify_bearer("Basic credentials")
    with pytest.raises(AuthenticationError):
        first.verify_bearer(None)


def test_unsigned_or_wrong_algorithm_token_is_rejected() -> None:
    service = DevTokenService(dev_settings())
    unsigned = jwt.encode(
        {"sub": "alice", "tenant_id": "tenant-a"}, key="", algorithm="none"
    )
    with pytest.raises(AuthenticationError):
        service.verify(unsigned)


def test_policy_requires_tenant_workspace_status_and_group() -> None:
    policy = BasicPolicyEngine()
    alice = Principal(user_id="alice", tenant_id="tenant-a", group_ids=["hr"])
    restricted = document(tenant_id="tenant-a", document_id="restricted", groups=["hr"])
    assert policy.authorize_read(alice, restricted, workspace_authorized=True)
    assert not policy.authorize_read(alice, restricted, workspace_authorized=False)
    assert not policy.authorize_read(
        alice,
        document(tenant_id="tenant-b", document_id="other", groups=["hr"]),
        workspace_authorized=True,
    )
    assert not policy.authorize_read(
        alice,
        document(tenant_id="tenant-a", document_id="failed", groups=["hr"], status="failed"),
        workspace_authorized=True,
    )
    assert not policy.authorize_read(
        alice,
        document(tenant_id="tenant-a", document_id="finance", groups=["finance"]),
        workspace_authorized=True,
    )


def test_tenant_visibility_is_explicit_and_restricted_empty_acl_fails_closed() -> None:
    policy = BasicPolicyEngine()
    principal = Principal(user_id="alice", tenant_id="tenant-a")
    tenant_visible = document(
        tenant_id="tenant-a", document_id="handbook", visibility="tenant"
    )
    ambiguous = document(tenant_id="tenant-a", document_id="ambiguous")
    assert policy.authorize_read(principal, tenant_visible, workspace_authorized=True)
    assert not policy.authorize_read(principal, ambiguous, workspace_authorized=True)


@pytest.mark.asyncio
async def test_fake_store_builds_prefilter_scope_for_two_tenants_and_groups() -> None:
    store = MemoryAuthorizationMetadataStore()
    await store.put_workspace_membership("tenant-a", "hr", "alice")
    await store.put_workspace_membership("tenant-a", "hr", "charlie")
    await store.put_workspace_membership("tenant-b", "hr", "bob")
    await store.put_document(
        document(tenant_id="tenant-a", document_id="a-hr", groups=["hr"])
    )
    await store.put_document(
        document(tenant_id="tenant-a", document_id="a-public", visibility="tenant")
    )
    await store.put_document(
        document(tenant_id="tenant-a", document_id="a-finance", groups=["finance"])
    )
    await store.put_document(
        document(tenant_id="tenant-b", document_id="b-hr", groups=["hr"])
    )

    alice = Principal(user_id="alice", tenant_id="tenant-a", group_ids=["hr"])
    scope = await store.authorization_scope(alice, "hr")
    assert [item.document_id for item in scope.documents] == ["a-hr", "a-public"]
    assert scope.to_auth_filter().allowed_document_ids == frozenset({"a-hr", "a-public"})
    assert "b-hr" not in scope.to_auth_filter().allowed_document_ids

    outsider = Principal(user_id="mallory", tenant_id="tenant-a", group_ids=["hr"])
    assert (await store.authorization_scope(outsider, "hr")).documents == ()
    assert await store.get_authorized_document(alice, "hr", "a-finance") is None
    assert not await store.delete_authorized_document(alice, "hr", "b-hr")
    assert await store.get_document("tenant-b", "hr", "b-hr") is not None


@pytest.mark.asyncio
async def test_acl_version_update_revokes_old_group_and_allows_new_group() -> None:
    store = MemoryAuthorizationMetadataStore()
    await store.put_workspace_membership("tenant-a", "hr", "alice")
    await store.put_workspace_membership("tenant-a", "hr", "charlie")
    await store.put_document(
        document(tenant_id="tenant-a", document_id="salary", groups=["hr"])
    )
    alice = Principal(user_id="alice", tenant_id="tenant-a", group_ids=["hr"])
    charlie = Principal(user_id="charlie", tenant_id="tenant-a", group_ids=["finance"])
    assert len((await store.authorization_scope(alice, "hr")).documents) == 1

    await store.put_document(
        document(
            tenant_id="tenant-a",
            document_id="salary",
            groups=["finance"],
            acl_version=2,
        )
    )
    assert (await store.authorization_scope(alice, "hr")).documents == ()
    updated = await store.authorization_scope(charlie, "hr")
    assert updated.documents[0].acl_version == 2
    assert updated.documents[0].index_version == "index-salary-a2"


def test_tool_authorization_requires_exact_server_verified_scope() -> None:
    policy = BasicPolicyEngine()
    principal = Principal(
        user_id="alice", tenant_id="tenant-a", scopes=["tools:search_docs"]
    )
    assert policy.authorize_tool(principal, "search_docs")
    assert not policy.authorize_tool(principal, "sql")
