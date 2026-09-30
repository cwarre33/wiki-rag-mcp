from langchain_core.language_models.fake_chat_models import FakeListChatModel

from wikirag.graph import NO_ANSWER, _parse_indices, ask, build_graph


def test_answers_from_graded_sources_and_strips_bad_citations(index):
    llm = FakeListChatModel(responses=["[1]", "It returns the consensus candidate [1] per [7]."])
    out = ask(build_graph(index, llm, k=2), "How does MBR pick an output?")
    assert out["answer"] == "It returns the consensus candidate [1] per ."
    assert len(out["sources"]) == 1


def test_rewrites_once_then_refuses_when_nothing_is_relevant(index):
    llm = FakeListChatModel(responses=["[]", "mbr decoding", "[]"])
    out = ask(build_graph(index, llm, k=2), "What is Cameron's favorite pizza?")
    assert out["answer"] == NO_ANSWER
    assert out["sources"] == []
    assert out["rewrites"] == 1
    assert out["query"] == "mbr decoding"


def test_rewrite_can_recover(index):
    llm = FakeListChatModel(responses=["[]", "faiss inner product", "[2]", "Exact search [1]."])
    out = ask(build_graph(index, llm, k=2), "how is similarity computed")
    assert out["answer"] == "Exact search [1]."


def test_parse_indices_is_forgiving():
    assert _parse_indices("Relevant: [3, 1, 1, 9]", 4) == [1, 3]
    assert _parse_indices("none of them", 4) == []
    assert _parse_indices("[]", 4) == []
