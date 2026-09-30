"""MCP server exposing the wiki to LLM agents: semantic search, full-page reads, and a grounded Q&A tool."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from typing import Any

from mcp.server import MCPServer

from .index import WikiIndex

MAX_K = 20


def build_server(index: WikiIndex, graph_factory: Callable[[], Any] | None = None) -> MCPServer:
    """`graph_factory` builds the LangGraph agent lazily, so search-only clients never need an LLM running."""
    server = MCPServer(
        "wiki-rag",
        instructions=(
            "Search and read a personal engineering wiki. Use search_wiki to find passages, get_page to read a "
            "whole page, and ask for a cited answer grounded in the wiki."
        ),
    )
    graph = None

    @server.tool()
    def search_wiki(query: str, k: int = 5) -> list[dict]:
        """Semantic search over wiki passages. Returns the top-k chunks with page path, heading, and score."""
        if not query.strip():
            raise ValueError("query must not be empty")
        return [h.to_dict() for h in index.search(query, k=max(1, min(k, MAX_K)))]

    @server.tool()
    def get_page(path: str) -> str:
        """Return the full markdown of one wiki page by its path, e.g. 'techniques/mbr-decoding.md'."""
        if path not in index.pages:
            raise ValueError(f"unknown page: {path}")
        return index.pages[path]

    @server.tool()
    def ask(question: str) -> dict:
        """Answer a question from the wiki with numbered citations; says so when the wiki has no answer."""
        nonlocal graph
        if graph_factory is None:
            raise RuntimeError("ask is disabled: no LLM configured")
        if graph is None:
            graph = graph_factory()
        out = graph.invoke({"question": question, "rewrites": 0})
        return {
            "answer": out["answer"],
            "sources": [{k: s[k] for k in ("path", "heading", "score")} for s in out.get("sources", [])],
        }

    @server.resource(
        "wiki://pages",
        name="pages",
        description="Every indexed page path and title",
        mime_type="application/json",
    )
    def list_pages() -> str:
        titles = {c.path: c.title for c in index.chunks}
        return json.dumps([{"path": p, "title": titles.get(p, p)} for p in sorted(index.pages)])

    return server


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the wiki MCP server over stdio.")
    ap.add_argument("--index", default=os.environ.get("WIKIRAG_INDEX", ".index"))
    ap.add_argument("--no-llm", action="store_true", help="disable the ask tool (search and read only)")
    args = ap.parse_args()
    index = WikiIndex.load(args.index)

    def factory():
        from .graph import build_graph

        return build_graph(index)

    build_server(index, None if args.no_llm else factory).run("stdio")


if __name__ == "__main__":
    main()
