"""Deterministic in-memory ingestion boundaries for offline tests."""

from app.ingestion.pipeline import StoredVersion


class MemoryIngestionMetadataStore:
    """Keep staged history and an atomic active pointer in memory."""

    def __init__(self) -> None:
        self.versions: dict[str, StoredVersion] = {}
        self.active_indexes: dict[tuple[str, str, str], str] = {}
        self.failed_indexes: set[str] = set()

    async def get_active(
        self, tenant_id: str, workspace_id: str, document_id: str
    ) -> StoredVersion | None:
        index_version = self.active_indexes.get((tenant_id, workspace_id, document_id))
        if index_version is None:
            return None
        return self.versions[index_version].model_copy(deep=True)

    async def stage(self, version: StoredVersion) -> None:
        self.versions[version.document.index_version] = version.model_copy(deep=True)

    async def activate(
        self, tenant_id: str, workspace_id: str, document_id: str, index_version: str
    ) -> StoredVersion:
        staged = self.versions[index_version]
        ready = staged.model_copy(
            update={"document": staged.document.model_copy(update={"status": "ready"})},
            deep=True,
        )
        self.versions[index_version] = ready
        self.active_indexes[(tenant_id, workspace_id, document_id)] = index_version
        return ready.model_copy(deep=True)

    async def mark_failed(self, index_version: str) -> None:
        staged = self.versions.get(index_version)
        if staged is not None:
            self.versions[index_version] = staged.model_copy(
                update={"document": staged.document.model_copy(update={"status": "failed"})},
                deep=True,
            )
        self.failed_indexes.add(index_version)


class MemoryIndexWriter:
    """Record isolated index versions and expose controlled failure points."""

    def __init__(self) -> None:
        self.versions: dict[str, StoredVersion] = {}
        self.write_calls = 0
        self.fail_write = False
        self.fail_validation = False

    async def write_version(self, version: StoredVersion) -> None:
        self.write_calls += 1
        self.versions[version.document.index_version] = version.model_copy(deep=True)
        if self.fail_write:
            raise RuntimeError("configured index write failure")

    async def validate_version(self, index_version: str) -> None:
        if index_version not in self.versions:
            raise RuntimeError("index version is missing")
        if self.fail_validation:
            raise RuntimeError("configured index validation failure")

    async def discard_version(self, index_version: str) -> None:
        self.versions.pop(index_version, None)
