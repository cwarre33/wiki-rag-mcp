import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from mcp import Client

from wikirag.graph import build_graph
from wikirag.server import build_server


def _payload(result):
    if result.structured_content is not None:
        return result.structured_content.get("result", result.structured_content)
    return json.loads(result.content[0].text)


async def test_lists_tools_and_searches(index):
    async with Client(build_server(index)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert names == {"search_wiki", "get_page", "ask"}
        hits = _payload(await client.call_tool("search_wiki", {"query": "FAISS inner product", "k": 1}))
        assert hits[0]["path"] == "tools/faiss.md"


async def test_get_page_and_unknown_page(index):
    async with Client(build_server(index)) as client:
        page = await client.call_tool("get_page", {"path": "techniques/mbr-decoding.md"})
        assert "average similarity" in page.content[0].text
        missing = await client.call_tool("get_page", {"path": "production-systems/secret.md"})
        assert missing.is_error


async def test_ask_is_disabled_without_an_llm(index):
    async with Client(build_server(index)) as client:
        assert (await client.call_tool("ask", {"question": "x"})).is_error


async def test_ask_returns_cited_answer(index):
    llm = FakeListChatModel(responses=["[1]", "Consensus candidate [1]."])
    async with Client(build_server(index, lambda: build_graph(index, llm, k=2))) as client:
        out = _payload(await client.call_tool("ask", {"question": "How does MBR work?"}))
        assert out["answer"] == "Consensus candidate [1]."
        assert out["sources"][0]["path"] == "techniques/mbr-decoding.md"


async def test_pages_resource_lists_only_indexed_pages(index):
    async with Client(build_server(index)) as client:
        res = await client.read_resource("wiki://pages")
        pages = json.loads(res.contents[0].text)
        assert [p["path"] for p in pages] == ["techniques/mbr-decoding.md", "tools/faiss.md"]


@pytest.mark.parametrize("query", ["", "   "])
async def test_empty_query_is_rejected(index, query):
    async with Client(build_server(index)) as client:
        assert (await client.call_tool("search_wiki", {"query": query})).is_error
