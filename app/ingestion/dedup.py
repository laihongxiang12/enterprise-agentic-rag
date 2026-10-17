"""Canonical text and exact-dedup helpers."""

import hashlib
import re
import unicodedata
from collections.abc import Iterable


def canonicalize_text(text: str) -> str:
    """Normalize representation without changing document meaning."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.splitlines())
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def exact_content_hash(text: str) -> str:
    """Return a SHA256 hash over canonical UTF-8 text."""
    return hashlib.sha256(canonicalize_text(text).encode("utf-8")).hexdigest()


def normalize_acl_groups(groups: Iterable[str]) -> tuple[str, ...]:
    """Create a deterministic ACL representation for comparison and storage."""
    return tuple(sorted({group.strip() for group in groups if group.strip()}))
