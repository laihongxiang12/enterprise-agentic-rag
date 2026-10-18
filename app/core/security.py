"""Verified development authentication and deterministic authorization policy."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import jwt
from jwt import InvalidTokenError

from app.core.config import Settings
from app.core.dependencies import VerifiedIdentity, build_principal
from app.schemas.documents import DocumentMeta, Principal
from app.tools.policy import DEFAULT_TOOL_POLICY, ToolPolicy

_MAX_AUTHORIZATION_LENGTH = 8_192
_MAX_TOKEN_LENGTH = 8_000
_MAX_CLAIM_ITEMS = 128
_MAX_IDENTITY_LENGTH = 128
_MAX_ATTRIBUTE_LENGTH = 256


class AuthenticationError(ValueError):
    """Authentication failed without exposing token or secret material."""


class DevTokenService:
    """Issue and verify short-lived local-only HS256 development tokens."""

    def __init__(self, settings: Settings) -> None:
        if not settings.dev_auth_enabled or settings.dev_auth_secret is None:
            raise ValueError("development authentication is disabled")
        self._secret = settings.dev_auth_secret.get_secret_value()
        self._issuer = settings.dev_auth_issuer
        self._audience = settings.dev_auth_audience

    def issue(self, identity: VerifiedIdentity, *, ttl_seconds: int = 900) -> str:
        if not 1 <= ttl_seconds <= 3_600:
            raise ValueError("development token TTL must be between 1 and 3600 seconds")
        now = datetime.now(UTC)
        payload = {
            "sub": identity.user_id,
            "tenant_id": identity.tenant_id,
            "groups": list(identity.group_ids),
            "scopes": list(identity.scopes),
            "iss": self._issuer,
            "aud": self._audience,
            "iat": now,
            "exp": now + timedelta(seconds=ttl_seconds),
        }
        return jwt.encode(payload, self._secret, algorithm="HS256")

    def verify(self, token: str) -> Principal:
        if not isinstance(token, str) or not token or len(token) > _MAX_TOKEN_LENGTH:
            raise AuthenticationError("invalid development bearer token")
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=["HS256"],
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["sub", "tenant_id", "iss", "aud", "iat", "exp"]},
            )
            identity = VerifiedIdentity(
                user_id=_required_claim(claims, "sub"),
                tenant_id=_required_claim(claims, "tenant_id"),
                group_ids=_string_tuple(claims.get("groups", []), "groups"),
                scopes=_string_tuple(claims.get("scopes", []), "scopes"),
            )
        except (InvalidTokenError, TypeError, ValueError) as exc:
            raise AuthenticationError("invalid development bearer token") from exc
        return build_principal(identity)

    def verify_bearer(self, authorization: str | None) -> Principal:
        if authorization is None:
            raise AuthenticationError("bearer token is required")
        if len(authorization) > _MAX_AUTHORIZATION_LENGTH:
            raise AuthenticationError("authorization header is too long")
        scheme, separator, token = authorization.partition(" ")
        if separator != " " or scheme.lower() != "bearer" or not token.strip():
            raise AuthenticationError("authorization must use Bearer scheme")
        return self.verify(token.strip())


class BasicPolicyEngine:
    """Fail-closed tenant, workspace, status, and group authorization rules."""

    def __init__(self, tool_policy: ToolPolicy | None = None) -> None:
        self._tool_policy = tool_policy or DEFAULT_TOOL_POLICY

    def authorize_read(
        self,
        principal: Principal,
        document: DocumentMeta,
        *,
        workspace_authorized: bool,
    ) -> bool:
        if principal.tenant_id != document.tenant_id or not workspace_authorized:
            return False
        if document.status != "ready":
            return False
        if document.visibility == "tenant":
            return True
        if not document.acl_groups:
            return False
        return bool(set(principal.group_ids).intersection(document.acl_groups))

    def authorize_tool(self, principal: Principal, tool_name: str) -> bool:
        return self._tool_policy.authorize(principal, tool_name)


def audit_safe_details(_details: Mapping[str, object]) -> dict[str, object]:
    """Drop free-form audit payloads so prompts, content, and keys cannot persist.

    Stable event identifiers, event type/action, and outcome live in dedicated
    audit columns. The JSON details column is deliberately kept empty.
    """

    return {}


def _required_claim(claims: dict[str, object], name: str) -> str:
    value = claims.get(name)
    if not isinstance(value, str):
        raise ValueError(f"claim {name} must be a non-empty string")
    return _safe_claim_value(value, name, max_length=_MAX_IDENTITY_LENGTH)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"claim {name} must be a string list")
    if len(value) > _MAX_CLAIM_ITEMS:
        raise ValueError(f"claim {name} contains too many items")
    return tuple(
        sorted({_safe_claim_value(item, name, max_length=_MAX_ATTRIBUTE_LENGTH) for item in value})
    )


def _safe_claim_value(value: str, name: str, *, max_length: int) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > max_length:
        raise ValueError(f"claim {name} has an invalid length")
    if any(not character.isprintable() for character in normalized):
        raise ValueError(f"claim {name} contains control characters")
    return normalized
