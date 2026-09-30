"""Stateful RAG agent built with LangGraph.

    retrieve -> grade --(relevant docs)--> generate -> END
                  \\--(none, retries left)--> rewrite -> retrieve
                  \\--(none, no retries)----> refuse -> END

The grader and the refusal path are the guardrails: the model only answers from chunks it judged relevant,
and citations that point at sources it was not given are stripped before the answer is returned.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .index import Hit, WikiIndex

NO_ANSWER = "The wiki does not cover this."
MAX_REWRITES = 1


class RAGState(TypedDict, total=False):
    question: str
    query: str
    hits: list[Hit]
    relevant: list[Hit]
    rewrites: int
    answer: str
    sources: list[dict]


def default_llm() -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(model=os.environ.get("WIKIRAG_MODEL", "llama3.2:3b"), temperature=0)


def _text(msg) -> str:
    return msg.content if isinstance(msg.content, str) else str(msg.content)


def _parse_indices(raw: str, n: int) -> list[int]:
    """Accept a JSON list like [1, 3] anywhere in the reply; ignore out-of-range values."""
    m = re.search(r"\[[\d,\s]*\]", raw)
    if not m:
        return []
    try:
        vals = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return sorted({v for v in vals if isinstance(v, int) and 1 <= v <= n})


def _format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, 1))


def build_graph(index: WikiIndex, llm: BaseChatModel | None = None, k: int = 5):
    llm = llm or default_llm()

    def retrieve(state: RAGState) -> RAGState:
        return {"hits": index.search(state.get("query") or state["question"], k=k)}

    def grade(state: RAGState) -> RAGState:
        hits = state["hits"]
        prompt = (
            f"Question: {state['question']}\n\nSources:\n{_format_sources(hits)}\n\n"
            "Which sources contain information that helps answer the question? "
            "Reply with only a JSON list of source numbers, e.g. [1, 3]. Reply [] if none do."
        )
        reply = _text(
            llm.invoke([SystemMessage("You judge retrieval relevance. Be strict."), HumanMessage(prompt)])
        )
        return {"relevant": [hits[i - 1] for i in _parse_indices(reply, len(hits))]}

    def rewrite(state: RAGState) -> RAGState:
        reply = _text(
            llm.invoke(
                [
                    HumanMessage(
                        "Rewrite this question as a short keyword search query for a personal engineering wiki. "
                        f"Reply with the query only.\n\nQuestion: {state['question']}"
                    )
                ]
            )
        )
        return {"query": reply.strip().strip('"')[:200], "rewrites": state.get("rewrites", 0) + 1}

    def generate(state: RAGState) -> RAGState:
        rel = state["relevant"]
        reply = _text(
            llm.invoke(
                [
                    SystemMessage(
                        "Answer using only the numbered sources. Cite each claim with its source number in brackets, "
                        f"e.g. [1]. If the sources do not answer the question, reply exactly: {NO_ANSWER}"
                    ),
                    HumanMessage(f"Sources:\n{_format_sources(rel)}\n\nQuestion: {state['question']}"),
                ]
            )
        ).strip()
        # Guardrail: drop citations to sources the model was not given.
        answer = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(rel) else "", reply)
        return {"answer": answer, "sources": [h.to_dict() for h in rel]}

    def refuse(state: RAGState) -> RAGState:
        return {"answer": NO_ANSWER, "sources": []}

    def after_grade(state: RAGState) -> str:
        if state["relevant"]:
            return "generate"
        return "rewrite" if state.get("rewrites", 0) < MAX_REWRITES else "refuse"

    g = StateGraph(RAGState)
    for name, fn in [
        ("retrieve", retrieve),
        ("grade", grade),
        ("rewrite", rewrite),
        ("generate", generate),
        ("refuse", refuse),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, ["generate", "rewrite", "refuse"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("refuse", END)
    return g.compile()


def ask(graph, question: str) -> RAGState:
    return graph.invoke({"question": question, "rewrites": 0})


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the wiki a question.")
    ap.add_argument("question")
    ap.add_argument("--index", default=os.environ.get("WIKIRAG_INDEX", ".index"))
    args = ap.parse_args()
    out = ask(build_graph(WikiIndex.load(args.index)), args.question)
    print(out["answer"])
    for i, s in enumerate(out.get("sources", []), 1):
        print(f"  [{i}] {s['path']} — {s['heading']}")


if __name__ == "__main__":
    main()
