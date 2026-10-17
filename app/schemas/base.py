"""Shared Pydantic behavior for application contracts."""

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Reject unknown fields instead of silently accepting unsafe input."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
