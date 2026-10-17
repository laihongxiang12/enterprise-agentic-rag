"""Verified development authentication and deterministic authorization policy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from jwt import InvalidTokenError

from app.core.config import Settings
from app.core.dependencies import VerifiedIdentity, build_principal
from app.schemas.documents import DocumentMeta, Principal


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
        scheme, separator, token = authorization.partition(" ")
        if separator != " " or scheme.lower() != "bearer" or not token.strip():
            raise AuthenticationError("authorization must use Bearer scheme")
        return self.verify(token.strip())


class BasicPolicyEngine:
    """Fail-closed tenant, workspace, status, and group authorization rules."""

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
        return f"tools:{tool_name}" in principal.scopes


def _required_claim(claims: dict[str, object], name: str) -> str:
    value = claims.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"claim {name} must be a non-empty string")
    return value.strip()


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"claim {name} must be a string list")
    return tuple(sorted({item.strip() for item in value if item.strip()}))
