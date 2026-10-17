"""Parent-child chunking behavior and stable lineage tests."""

from pathlib import Path

import pytest

from app.ingestion.chunking import ChunkContext, ChunkingConfig, ParentChildChunker
from app.ingestion.dedup import exact_content_hash
from app.ingestion.loaders import load_document
from app.ingestion.models import LoadedDocument, PageBlock

FIXTURES = Path(__file__).parents[1] / "fixtures" / "documents"


def context_for(document: LoadedDocument, *, version: int = 1) -> ChunkContext:
    return ChunkContext(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        document_id="policy-1",
        document_version=version,
        content_hash=exact_content_hash(document.text),
        acl_groups=["employees"],
        acl_version=1,
        index_version=f"index-v{version}",
    )


def test_short_document_has_stable_parent_and_child_ids() -> None:
    document = load_document(FIXTURES / "it_policy_en.md")
    chunker = ParentChildChunker(ChunkingConfig(parent_size=500, child_size=200))
    first = chunker.chunk(document, context_for(document))
    second = chunker.chunk(document, context_for(document))
    assert [chunk.parent_id for chunk in first.parents] == [
        chunk.parent_id for chunk in second.parents
    ]
    assert [chunk.chunk_id for chunk in first.children] == [
        chunk.chunk_id for chunk in second.children
    ]
    parent_ids = {parent.parent_id for parent in first.parents}
    assert all(child.parent_id in parent_ids for child in first.children)


def test_long_chinese_paragraph_is_bounded_and_split() -> None:
    text = "值班人员必须确认告警，记录工单并保护访问密钥。" * 80  # noqa: RUF001
    document = LoadedDocument(
        text=text,
        title="超长值班制度",
        source_uri="fixture:///long-zh.md",
        page_blocks=[PageBlock(text=text, heading_path=["值班制度"])],
        metadata={"source_type": "markdown"},
    )
    chunker = ParentChildChunker(
        ChunkingConfig(parent_size=240, child_size=80, child_overlap=12)
    )
    chunks = chunker.chunk(document, context_for(document))
    assert len(chunks.parents) > 2
    assert len(chunks.children) > len(chunks.parents)
    assert max(len(chunk.content) for chunk in chunks.parents) <= 240
    assert max(len(chunk.content) for chunk in chunks.children) <= 80
    assert all(chunk.heading_path == ["值班制度"] for chunk in chunks.children)


def test_pdf_page_and_source_lineage_reaches_every_child() -> None:
    document = load_document(FIXTURES / "it_oncall_zh.pdf")
    chunks = ParentChildChunker().chunk(document, context_for(document))
    assert chunks.children
    assert all(chunk.page == 1 for chunk in chunks.children)
    assert all(chunk.source_uri == document.source_uri for chunk in chunks.children)
    assert all(chunk.document_version == 1 for chunk in chunks.children)
    assert all(chunk.acl_groups == ["employees"] for chunk in chunks.children)


def test_document_version_changes_stable_ids() -> None:
    document = load_document(FIXTURES / "it_policy_en.md")
    chunker = ParentChildChunker()
    version_one = chunker.chunk(document, context_for(document, version=1))
    version_two = chunker.chunk(document, context_for(document, version=2))
    assert version_one.children[0].chunk_id != version_two.children[0].chunk_id


def test_valid_boundary_config() -> None:
    config = ChunkingConfig(parent_size=100, child_size=100, child_overlap=0)
    assert ParentChildChunker(config).config == config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parent_size": 0},
        {"parent_size": 100, "child_size": 101},
        {"child_size": 20, "child_overlap": 20},
        {"child_overlap": -1},
    ],
)
def test_invalid_chunking_config_is_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)
