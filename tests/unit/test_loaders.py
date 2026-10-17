"""Safe document loader behavior and negative-path tests."""

from pathlib import Path

import pytest

from app.ingestion.loaders import load_document
from app.ingestion.models import DocumentLoadError, LoadErrorCode

FIXTURES = Path(__file__).parents[1] / "fixtures" / "documents"


def assert_load_error(code: LoadErrorCode, path: Path, **kwargs: object) -> None:
    with pytest.raises(DocumentLoadError) as caught:
        load_document(path, **kwargs)  # type: ignore[arg-type]
    assert caught.value.code is code


def test_loads_chinese_markdown_and_preserves_heading_paths() -> None:
    document = load_document(FIXTURES / "hr_policy_zh.md", content_type="text/markdown")

    assert document.title == "Northstar Tech 人事休假制度"
    assert "员工应至少提前三个工作日" in document.text
    assert document.page_blocks[1].heading_path == ["Northstar Tech 人事休假制度", "年假申请"]
    assert document.metadata["source_type"] == "markdown"
    assert len(str(document.metadata["content_sha256"])) == 64


def test_loads_second_markdown_fixture() -> None:
    document = load_document(FIXTURES / "it_policy_en.md")
    assert document.title == "Northstar Access Policy"
    assert "time-limited credentials" in document.text


def test_loads_html_without_script_or_style_and_preserves_headings() -> None:
    document = load_document(FIXTURES / "finance_policy_zh.html", content_type="text/html")

    assert document.title == "Northstar 财务报销制度"
    assert "住宿费用必须附上合规发票" in document.text
    assert "window.evil" not in document.text
    assert "display: none" not in document.text
    assert document.page_blocks[-1].heading_path == ["差旅报销", "审批时限"]


def test_loads_second_html_fixture() -> None:
    document = load_document(FIXTURES / "security_policy_en.html")
    assert document.title == "Northstar Security Policy"
    assert "credential exposure" in document.text


@pytest.mark.parametrize(
    ("filename", "expected_text"),
    [
        ("it_oncall_zh.pdf", "十分钟内确认"),
        ("expense_policy_en.pdf", "Receipts are required"),
    ],
)
def test_loads_pdf_with_real_page_numbers(filename: str, expected_text: str) -> None:
    document = load_document(FIXTURES / filename, content_type="application/pdf")

    assert expected_text in document.text
    assert document.page_blocks[0].page == 1
    assert document.metadata["page_count"] == 1


def test_encrypted_pdf_has_stable_error() -> None:
    assert_load_error(LoadErrorCode.ENCRYPTED_PDF, FIXTURES / "encrypted_policy.pdf")


def test_damaged_pdf_has_stable_error() -> None:
    assert_load_error(LoadErrorCode.INVALID_DOCUMENT, FIXTURES / "broken.pdf")


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.md"
    empty.touch()
    assert_load_error(LoadErrorCode.EMPTY_FILE, empty)


def test_whitespace_only_markdown_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "blank.md"
    empty.write_text(" \n\t", encoding="utf-8")
    assert_load_error(LoadErrorCode.EMPTY_FILE, empty)


def test_oversized_file_is_rejected_before_parsing(tmp_path: Path) -> None:
    large = tmp_path / "large.md"
    large.write_bytes(b"a" * 33)
    assert_load_error(LoadErrorCode.FILE_TOO_LARGE, large, max_bytes=32)


def test_invalid_utf8_is_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.md"
    invalid.write_bytes(b"\xff\xfe\xfa")
    assert_load_error(LoadErrorCode.INVALID_ENCODING, invalid)


def test_declared_mime_mismatch_is_rejected() -> None:
    assert_load_error(
        LoadErrorCode.MIME_MISMATCH,
        FIXTURES / "hr_policy_zh.md",
        content_type="text/html",
    )


def test_pdf_signature_under_markdown_extension_is_rejected(tmp_path: Path) -> None:
    disguised = tmp_path / "disguised.md"
    disguised.write_bytes((FIXTURES / "expense_policy_en.pdf").read_bytes())
    assert_load_error(LoadErrorCode.MIME_MISMATCH, disguised)


def test_html_without_markup_is_rejected(tmp_path: Path) -> None:
    fake_html = tmp_path / "fake.html"
    fake_html.write_text("plain text only", encoding="utf-8")
    assert_load_error(LoadErrorCode.INVALID_DOCUMENT, fake_html)


def test_html_containing_only_script_has_no_extractable_text(tmp_path: Path) -> None:
    script_only = tmp_path / "script.html"
    script_only.write_text("<script>alert('x')</script>", encoding="utf-8")
    assert_load_error(LoadErrorCode.NO_EXTRACTABLE_TEXT, script_only)


def test_unsupported_extension_and_missing_file_are_distinct(tmp_path: Path) -> None:
    unsupported = tmp_path / "policy.txt"
    unsupported.write_text("content", encoding="utf-8")
    assert_load_error(LoadErrorCode.UNSUPPORTED_TYPE, unsupported)
    assert_load_error(LoadErrorCode.FILE_NOT_FOUND, tmp_path / "missing.md")


def test_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    target.write_text("# Safe target", encoding="utf-8")
    link = tmp_path / "link.md"
    link.symlink_to(target)
    assert_load_error(LoadErrorCode.UNSAFE_PATH, link)


def test_page_block_mutable_defaults_are_independent() -> None:
    first = load_document(FIXTURES / "it_policy_en.md")
    second = load_document(FIXTURES / "it_policy_en.md")
    first.page_blocks[0].heading_path.append("mutated")
    assert "mutated" not in second.page_blocks[0].heading_path
