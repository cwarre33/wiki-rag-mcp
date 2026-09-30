"""Load a markdown knowledge base, apply the publication filter, and split pages into chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Sections and tag fragments that stay out of the index even when a page is marked public.
REDACTED_SECTIONS = frozenset({"open-questions", "case-studies"})
REDACTED_TAGS = ("security", "osint", "disclosure", "ics", "shodan", "credential")

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_WIKILINK = re.compile(r"\[\[([^\]|]+?)(?:\|([^\]]+))?\]\]")
_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class Page:
    path: str  # relative to the corpus root, forward slashes
    title: str
    section: str
    tags: tuple[str, ...]
    body: str


@dataclass(frozen=True)
class Chunk:
    id: str
    path: str
    title: str
    heading: str
    text: str
    meta: dict = field(default_factory=dict, compare=False)


def _lenient_frontmatter(block: str) -> dict:
    """Line-wise `key: value` parse for Obsidian-style frontmatter that is not valid YAML,
    e.g. `related: [[a.md]], [[b.md]]`."""
    fm: dict = {}
    for line in block.splitlines():
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val.startswith("[") and val.endswith("]") and not val.startswith("[["):
            fm[key] = [v.strip().strip("\"'") for v in val[1:-1].split(",") if v.strip()]
        else:
            fm[key] = val.strip("\"'")
    return fm


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    m = _FRONTMATTER.match(raw)
    if not m:
        return {}, raw
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        fm = _lenient_frontmatter(m.group(1))
    return (fm if isinstance(fm, dict) else {}), raw[m.end() :]


def is_publishable(fm: dict, section: str) -> bool:
    if fm.get("visibility") != "public" or section in REDACTED_SECTIONS:
        return False
    tags = [str(t).lower() for t in (fm.get("tags") or [])]
    return not any(r in t for t in tags for r in REDACTED_TAGS)


def clean_links(text: str) -> str:
    """Render [[path|label]] as label and [[path]] as the page stem so embeddings see words, not paths."""

    def repl(m: re.Match) -> str:
        if m.group(2):
            return m.group(2)
        return Path(m.group(1)).stem.replace("-", " ")

    return _WIKILINK.sub(repl, text)


def load_pages(root: str | Path, public_only: bool = True) -> list[Page]:
    root = Path(root)
    pages = []
    for f in sorted(root.rglob("*.md")):
        rel = f.relative_to(root).as_posix()
        if rel in {"index.md", "log.md"}:
            continue
        fm, body = parse_frontmatter(f.read_text(encoding="utf-8"))
        section = rel.split("/")[0] if "/" in rel else ""
        if public_only and not is_publishable(fm, section):
            continue
        title = str(fm.get("title") or f.stem).strip().strip('"')
        tags = tuple(str(t) for t in (fm.get("tags") or []))
        pages.append(Page(path=rel, title=title, section=section, tags=tags, body=body.strip()))
    return pages


def _windows(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # Prefer to break on a paragraph or line boundary inside the window.
        if end < len(text):
            cut = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end))
            if cut > start + max_chars // 2:
                end = cut
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [w for w in out if w]


def chunk_page(page: Page, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    """Split on H1–H3 headings, then window long sections. Each chunk is prefixed with its page title."""
    body = clean_links(page.body)
    spans: list[tuple[str, str]] = []
    last, heading = 0, page.title
    for m in _HEADING.finditer(body):
        if m.start() > last:
            spans.append((heading, body[last : m.start()]))
        heading, last = m.group(2).strip(), m.start()
    spans.append((heading, body[last:]))

    chunks = []
    for heading, text in spans:
        text = text.strip()
        if len(text) < 40:
            continue
        for w in _windows(text, max_chars, overlap):
            cid = f"{page.path}#{len(chunks)}"
            chunks.append(
                Chunk(
                    id=cid,
                    path=page.path,
                    title=page.title,
                    heading=heading,
                    text=f"{page.title} — {heading}\n{w}",
                    meta={"section": page.section, "tags": list(page.tags)},
                )
            )
    return chunks


def chunk_pages(pages: list[Page], **kw) -> list[Chunk]:
    return [c for p in pages for c in chunk_page(p, **kw)]
