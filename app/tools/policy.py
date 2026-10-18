"""Fail-closed policy for tools exposed to the agent runtime."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from app.schemas.documents import Principal

_TOOL_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_DANGEROUS_TOOL_NAMES = frozenset({"filesystem_write", "python", "shell", "sql", "web"})


@dataclass(frozen=True, slots=True)
class ToolRegistration:
    """One server-controlled tool registration.

    A model response can request a name, but it cannot add a registration or
    change the required scope.
    """

    name: str
    required_scope: str
    read_only: bool = True

    def __post_init__(self) -> None:
        if _TOOL_NAME.fullmatch(self.name) is None:
            raise ValueError("tool name must be a normalized identifier")
        if self.required_scope != f"tools:{self.name}":
            raise ValueError("tool scope must be bound to the registered name")
        if self.name in _DANGEROUS_TOOL_NAMES or not self.read_only:
            raise ValueError("dangerous tools cannot be registered")


class ToolPolicy:
    """Authorize only exact names in an immutable server-side allowlist."""

    def __init__(self, registrations: tuple[ToolRegistration, ...] = ()) -> None:
        indexed: dict[str, ToolRegistration] = {}
        for registration in registrations:
            if registration.name in indexed:
                raise ValueError("duplicate tool registration")
            indexed[registration.name] = registration
        self._registrations: Mapping[str, ToolRegistration] = MappingProxyType(indexed)

    @property
    def registered_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._registrations))

    def authorize(self, principal: Principal, requested_name: str) -> bool:
        if not isinstance(requested_name, str) or _TOOL_NAME.fullmatch(requested_name) is None:
            return False
        registration = self._registrations.get(requested_name)
        return bool(
            registration is not None
            and registration.read_only
            and registration.required_scope in principal.scopes
        )


DEFAULT_TOOL_POLICY = ToolPolicy(
    (ToolRegistration(name="search_docs", required_scope="tools:search_docs"),)
)
