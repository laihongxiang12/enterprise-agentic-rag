"""Bounded, offline loaders for PDF, Markdown, and HTML documents."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.ingestion.models import DocumentLoadError, LoadedDocument, LoadErrorCode, PageBlock

DEFAULT_MAX_BYTES = 10 * 1024 * 1024

_TYPE_BY_SUFFIX = {
    ".htm": "html",
    ".html": "html",
    ".markdown": "markdown",
    ".md": "markdown",
    ".pdf": "pdf",
}
_MIME_BY_TYPE = {
    "html": frozenset({"text/html", "application/xhtml+xml"}),
    "markdown": frozenset({"text/markdown", "text/x-markdown", "text/plain"}),
    "pdf": frozenset({"application/pdf"}),
}
_HTML_TAG_RE = re.compile(r"<[A-Za-z][^>]*>")
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def load_document(
    path: str | Path,
    *,
    content_type: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    source_uri: str | None = None,
) -> LoadedDocument:
    """Load one local document without remote access or embedded-code execution."""
    file_path = Path(path)
    if file_path.is_symlink():
        raise DocumentLoadError(LoadErrorCode.UNSAFE_PATH, "Symbolic links are not accepted")
    if not file_path.exists() or not file_path.is_file():
        raise DocumentLoadError(LoadErrorCode.FILE_NOT_FOUND, "Document file was not found")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")

    document_type = _TYPE_BY_SUFFIX.get(file_path.suffix.lower())
    if document_type is None:
        raise DocumentLoadError(LoadErrorCode.UNSUPPORTED_TYPE, "Unsupported document type")

    normalized_mime = _normalize_mime(content_type)
    if normalized_mime is not None and normalized_mime not in _MIME_BY_TYPE[document_type]:
        raise DocumentLoadError(
            LoadErrorCode.MIME_MISMATCH,
            "Declared content type does not match the file extension",
        )

    data = _read_bounded(file_path, max_bytes=max_bytes)
    if data.startswith(b"%PDF-") != (document_type == "pdf"):
        raise DocumentLoadError(
            LoadErrorCode.MIME_MISMATCH,
            "Document signature does not match the file extension",
        )

    resolved_source = source_uri or file_path.resolve(strict=True).as_uri()
    loaders: dict[str, Callable[[bytes, Path, str], LoadedDocument]] = {
        "html": _load_html,
        "markdown": _load_markdown,
        "pdf": _load_pdf,
    }
    return loaders[document_type](data, file_path, resolved_source)


def _normalize_mime(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    return content_type.partition(";")[0].strip().lower()


def _read_bounded(path: Path, *, max_bytes: int) -> bytes:
    try:
        with path.open("rb") as stream:
            data = stream.read(max_bytes + 1)
    except OSError as exc:
        raise DocumentLoadError(
            LoadErrorCode.INVALID_DOCUMENT, "Document could not be read"
        ) from exc
    if not data:
        raise DocumentLoadError(LoadErrorCode.EMPTY_FILE, "Document is empty")
    if len(data) > max_bytes:
        raise DocumentLoadError(LoadErrorCode.FILE_TOO_LARGE, "Document exceeds size limit")
    return data


def _decode_utf8(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            LoadErrorCode.INVALID_ENCODING, "Document must use valid UTF-8 encoding"
        ) from exc


def _metadata(data: bytes, file_path: Path, document_type: str) -> dict[str, str | int]:
    return {
        "content_sha256": hashlib.sha256(data).hexdigest(),
        "filename": file_path.name,
        "size_bytes": len(data),
        "source_type": document_type,
    }


def _load_markdown(data: bytes, file_path: Path, source_uri: str) -> LoadedDocument:
    text = _decode_utf8(data).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise DocumentLoadError(LoadErrorCode.EMPTY_FILE, "Markdown document has no content")

    title = file_path.stem
    heading_path: list[str] = []
    blocks: list[PageBlock] = []
    current_lines: list[str] = []
    current_path: list[str] = []

    def flush() -> None:
        block_text = "\n".join(current_lines).strip()
        if block_text:
            blocks.append(PageBlock(text=block_text, heading_path=current_path.copy()))
        current_lines.clear()

    for line in text.splitlines():
        match = _MARKDOWN_HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            if title == file_path.stem:
                title = heading
            del heading_path[level - 1 :]
            heading_path.append(heading)
            current_path = heading_path.copy()
        current_lines.append(line)
    flush()

    return LoadedDocument(
        text=text,
        title=title,
        source_uri=source_uri,
        page_blocks=blocks,
        metadata=_metadata(data, file_path, "markdown"),
    )


class _SafeHTMLExtractor(HTMLParser):
    """Extract visible text and headings; script/style bodies are discarded."""

    _SKIPPED_TAGS = frozenset({"script", "style"})
    _BLOCK_TAGS = frozenset({"article", "blockquote", "div", "li", "main", "p", "section"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[PageBlock] = []
        self.title_parts: list[str] = []
        self._buffer: list[str] = []
        self._heading_path: list[str] = []
        self._pending_heading_level: int | None = None
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in self._SKIPPED_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in self._BLOCK_TAGS or tag == "br" or re.fullmatch(r"h[1-6]", tag):
            self._flush()
        if re.fullmatch(r"h[1-6]", tag):
            self._pending_heading_level = int(tag[1])

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIPPED_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in self._BLOCK_TAGS or re.fullmatch(r"h[1-6]", tag):
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        normalized = " ".join(data.split())
        if normalized:
            self._buffer.append(normalized)

    def _flush(self) -> None:
        text = " ".join(self._buffer).strip()
        self._buffer.clear()
        if not text:
            return
        if self._pending_heading_level is not None:
            level = self._pending_heading_level
            del self._heading_path[level - 1 :]
            self._heading_path.append(text)
            rendered = f"{'#' * level} {text}"
            self._pending_heading_level = None
        else:
            rendered = text
        self.blocks.append(PageBlock(text=rendered, heading_path=self._heading_path.copy()))

    def finish(self) -> None:
        self._flush()


def _load_html(data: bytes, file_path: Path, source_uri: str) -> LoadedDocument:
    raw_html = _decode_utf8(data)
    if _HTML_TAG_RE.search(raw_html) is None:
        raise DocumentLoadError(LoadErrorCode.INVALID_DOCUMENT, "HTML document has no tags")
    parser = _SafeHTMLExtractor()
    try:
        parser.feed(raw_html)
        parser.close()
        parser.finish()
    except Exception as exc:
        raise DocumentLoadError(LoadErrorCode.INVALID_DOCUMENT, "HTML could not be parsed") from exc
    if not parser.blocks:
        raise DocumentLoadError(
            LoadErrorCode.NO_EXTRACTABLE_TEXT, "HTML has no extractable visible text"
        )

    text = "\n\n".join(block.text for block in parser.blocks)
    html_title = " ".join(" ".join(parser.title_parts).split())
    title = html_title or next(
        (block.text.lstrip("# ") for block in parser.blocks if block.text.startswith("#")),
        file_path.stem,
    )
    return LoadedDocument(
        text=text,
        title=title,
        source_uri=source_uri,
        page_blocks=parser.blocks,
        metadata=_metadata(data, file_path, "html"),
    )


def _load_pdf(data: bytes, file_path: Path, source_uri: str) -> LoadedDocument:
    try:
        reader = PdfReader(BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise DocumentLoadError(LoadErrorCode.ENCRYPTED_PDF, "Encrypted PDFs are not accepted")
        blocks = [
            PageBlock(text=text, page=page_number)
            for page_number, page in enumerate(reader.pages, start=1)
            if (text := (page.extract_text() or "").strip())
        ]
    except DocumentLoadError:
        raise
    except (PdfReadError, OSError, ValueError) as exc:
        raise DocumentLoadError(LoadErrorCode.INVALID_DOCUMENT, "PDF could not be parsed") from exc
    if not blocks:
        raise DocumentLoadError(
            LoadErrorCode.NO_EXTRACTABLE_TEXT,
            "PDF has no extractable text; OCR is not supported",
        )

    metadata_title = None
    if reader.metadata is not None:
        metadata_title = reader.metadata.title
    title = metadata_title.strip() if metadata_title and metadata_title.strip() else file_path.stem
    return LoadedDocument(
        text="\n\n".join(block.text for block in blocks),
        title=title,
        source_uri=source_uri,
        page_blocks=blocks,
        metadata={**_metadata(data, file_path, "pdf"), "page_count": len(reader.pages)},
    )
