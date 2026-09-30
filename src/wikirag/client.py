"""Minimal MCP client: launches the server over stdio, lists its tools, and runs one search."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from mcp import Client, StdioServerParameters


async def run(query: str, index: str, k: int) -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "wikirag.server", "--index", index, "--no-llm"]
    )
    async with Client(params) as client:
        tools = await client.list_tools()
        print("tools:", ", ".join(t.name for t in tools.tools))
        result = await client.call_tool("search_wiki", {"query": query, "k": k})
        hits = (
            result.structured_content["result"]
            if result.structured_content
            else json.loads(result.content[0].text)
        )
        for h in hits:
            print(f"{h['score']:.3f}  {h['path']}  —  {h['heading']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Query the wiki MCP server from a client.")
    ap.add_argument("query")
    ap.add_argument("--index", default=".index")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    asyncio.run(run(args.query, args.index, args.k))


if __name__ == "__main__":
    main()
