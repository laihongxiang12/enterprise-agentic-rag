"""Deduplication and atomic version activation tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.ingestion.fakes import MemoryIndexWriter, MemoryIngestionMetadataStore
from app.ingestion.loaders import load_document
from app.ingestion.models import LoadedDocument, PageBlock
from app.ingestion.pipeline import IngestionFailure, IngestionPipeline, IngestionRequest

FIXTURES = Path(__file__).parents[1] / "fixtures" / "documents"


def request(*, acl_groups: list[str] | None = None) -> IngestionRequest:
    return IngestionRequest(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        document_id="hr-policy",
        acl_groups=["employees"] if acl_groups is None else acl_groups,
    )


@pytest.mark.parametrize("groups", [[], ["  "]])
def test_empty_acl_is_rejected_instead_of_becoming_public(groups: list[str]) -> None:
    with pytest.raises(ValidationError):
        request(acl_groups=groups)


def changed_document(original: LoadedDocument, suffix: str) -> LoadedDocument:
    text = f"{original.text}\n\n{suffix}"
    blocks = [*original.page_blocks, PageBlock(text=suffix, heading_path=["更新"])]
    return original.model_copy(update={"text": text, "page_blocks": blocks}, deep=True)


@pytest.fixture
def boundaries() -> tuple[MemoryIngestionMetadataStore, MemoryIndexWriter, IngestionPipeline]:
    store = MemoryIngestionMetadataStore()
    writer = MemoryIndexWriter()
    return store, writer, IngestionPipeline(store, writer)


@pytest.mark.asyncio
async def test_duplicate_import_does_not_add_chunks_or_write_again(
    boundaries: tuple[MemoryIngestionMetadataStore, MemoryIndexWriter, IngestionPipeline],
) -> None:
    store, writer, pipeline = boundaries
    document = load_document(FIXTURES / "hr_policy_zh.md")
    first = await pipeline.ingest(document, request())
    second = await pipeline.ingest(document, request(acl_groups=["employees", "employees"]))
    assert first.outcome == "indexed"
    assert second.outcome == "deduplicated"
    assert first.document.index_version == second.document.index_version
    assert writer.write_calls == 1
    assert len(writer.versions) == 1
    assert len(store.versions) == 1


@pytest.mark.asyncio
async def test_canonical_line_endings_are_exact_duplicates(
    boundaries: tuple[MemoryIngestionMetadataStore, MemoryIndexWriter, IngestionPipeline],
) -> None:
    _store, writer, pipeline = boundaries
    document = load_document(FIXTURES / "it_policy_en.md")
    await pipeline.ingest(document, request())
    crlf = document.model_copy(
        update={
            "text": document.text.replace("\n", "\r\n"),
            "page_blocks": [
                block.model_copy(update={"text": block.text.replace("\n", "\r\n")})
                for block in document.page_blocks
            ],
        },
        deep=True,
    )
    result = await pipeline.ingest(crlf, request())
    assert result.outcome == "deduplicated"
    assert writer.write_calls == 1


@pytest.mark.asyncio
async def test_content_update_activates_only_new_document_version(
    boundaries: tuple[MemoryIngestionMetadataStore, MemoryIndexWriter, IngestionPipeline],
) -> None:
    store, writer, pipeline = boundaries
    original = load_document(FIXTURES / "hr_policy_zh.md")
    first = await pipeline.ingest(original, request())
    changed = changed_document(original, "新增：年假可撤回。")  # noqa: RUF001
    second = await pipeline.ingest(changed, request())
    active = await store.get_active("tenant-a", "workspace-a", "hr-policy")
    assert first.document.version == 1
    assert second.outcome == "indexed"
    assert second.document.version == 2
    assert second.document.acl_version == 1
    assert first.document.index_version in writer.versions
    assert second.document.index_version in writer.versions
    assert active is not None
    assert active.document.index_version == second.document.index_version
    assert {child.document_version for child in active.chunks.children} == {2}


@pytest.mark.asyncio
async def test_acl_only_update_reindexes_without_changing_content_version(
    boundaries: tuple[MemoryIngestionMetadataStore, MemoryIndexWriter, IngestionPipeline],
) -> None:
    store, writer, pipeline = boundaries
    document = load_document(FIXTURES / "hr_policy_zh.md")
    first = await pipeline.ingest(document, request(acl_groups=["employees"]))
    first_chunks = writer.versions[first.document.index_version].chunks.children
    old_ids = {child.chunk_id for child in first_chunks}
    second = await pipeline.ingest(document, request(acl_groups=["hr-admins"]))
    active = await store.get_active("tenant-a", "workspace-a", "hr-policy")
    assert second.outcome == "acl_updated"
    assert second.document.version == 1
    assert second.document.acl_version == 2
    assert second.document.index_version != first.document.index_version
    assert active is not None
    assert active.document.acl_groups == ["hr-admins"]
    assert {child.acl_groups[0] for child in active.chunks.children} == {"hr-admins"}
    assert {child.chunk_id for child in active.chunks.children} == old_ids
    assert writer.write_calls == 2


@pytest.mark.asyncio
async def test_validation_failure_keeps_old_version_active_and_discards_new_index(
    boundaries: tuple[MemoryIngestionMetadataStore, MemoryIndexWriter, IngestionPipeline],
) -> None:
    store, writer, pipeline = boundaries
    original = load_document(FIXTURES / "hr_policy_zh.md")
    first = await pipeline.ingest(original, request())
    writer.fail_validation = True
    with pytest.raises(IngestionFailure):
        await pipeline.ingest(changed_document(original, "不会激活的内容"), request())
    active = await store.get_active("tenant-a", "workspace-a", "hr-policy")
    assert active is not None
    assert active.document.index_version == first.document.index_version
    assert active.document.version == 1
    assert len(writer.versions) == 1
    assert len(store.failed_indexes) == 1
    failed_index = next(iter(store.failed_indexes))
    assert store.versions[failed_index].document.status == "failed"


@pytest.mark.asyncio
async def test_initial_write_failure_has_no_active_version(
    boundaries: tuple[MemoryIngestionMetadataStore, MemoryIndexWriter, IngestionPipeline],
) -> None:
    store, writer, pipeline = boundaries
    writer.fail_write = True
    document = load_document(FIXTURES / "hr_policy_zh.md")
    with pytest.raises(IngestionFailure):
        await pipeline.ingest(document, request())
    assert await store.get_active("tenant-a", "workspace-a", "hr-policy") is None
    assert not writer.versions
    assert len(store.failed_indexes) == 1


@pytest.mark.asyncio
async def test_page_lineage_survives_pipeline(
    boundaries: tuple[MemoryIngestionMetadataStore, MemoryIndexWriter, IngestionPipeline],
) -> None:
    store, _writer, pipeline = boundaries
    pdf = load_document(FIXTURES / "expense_policy_en.pdf")
    result = await pipeline.ingest(pdf, request())
    active = await store.get_active("tenant-a", "workspace-a", "hr-policy")
    assert result.document.source_type == "pdf"
    assert active is not None
    assert {child.page for child in active.chunks.children} == {1}
    assert all(child.parent_id for child in active.chunks.children)
    assert all(child.source_uri == pdf.source_uri for child in active.chunks.children)
