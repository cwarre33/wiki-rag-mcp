"""Evaluate the wiki RAG pipeline against eval/golden.jsonl.

Retrieval metrics (no LLM needed):
  hit@k           a reference page appears in the top-k chunks
  MRR             reciprocal rank of the first chunk from a reference page
  snippet recall  share of reference passages found verbatim (markdown-normalized) in the retrieved text

Generation metrics (--llm; Ragas with an LLM judge, default Ollama):
  faithfulness, answer relevancy (Ragas ResponseRelevancy), LLM context recall,
  plus refusal accuracy on questions the wiki cannot answer.

Usage:
  python eval/run_eval.py --index .index                # retrieval only
  python eval/run_eval.py --index .index --llm          # full run; needs `ollama serve` and WIKIRAG_MODEL pulled
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import sys
from pathlib import Path

from wikirag.index import WikiIndex

HERE = Path(__file__).parent


def normalize(s: str) -> str:
    s = re.sub(r"[*_`#>|]", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def retrieval_metrics(index: WikiIndex, golden: list[dict], k: int) -> tuple[dict, list[dict]]:
    rows = []
    for g in golden:
        if not g["reference_pages"]:
            continue
        hits = index.search(g["question"], k=k)
        paths = [h.chunk.path for h in hits]
        rank = next((i for i, p in enumerate(paths, 1) if p in g["reference_pages"]), None)
        blob = normalize(" ".join(h.chunk.text for h in hits))
        found = [normalize(c) in blob for c in g["reference_contexts"]]
        rows.append(
            {
                "question": g["question"],
                "hit": rank is not None,
                "rr": 1 / rank if rank else 0.0,
                "snippet_recall": sum(found) / len(found) if found else None,
                "top": paths[:3],
            }
        )
    recalls = [r["snippet_recall"] for r in rows if r["snippet_recall"] is not None]
    summary = {
        f"hit@{k}": statistics.mean(r["hit"] for r in rows),
        "mrr": statistics.mean(r["rr"] for r in rows),
        "snippet_recall": statistics.mean(recalls) if recalls else None,
        "questions": len(rows),
    }
    return summary, rows


def generation_metrics(index: WikiIndex, golden: list[dict], model: str | None) -> tuple[dict, list[dict]]:
    import os

    from langchain_core.embeddings import Embeddings
    from langchain_ollama import ChatOllama
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy

    from wikirag.graph import NO_ANSWER, ask, build_graph

    model = model or os.environ.get("WIKIRAG_MODEL", "llama3.2:3b")
    llm = ChatOllama(model=model, temperature=0)
    graph = build_graph(index, llm)

    class IndexEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return index.embedder.encode(list(texts)).tolist()

        def embed_query(self, text):
            return index.embedder.encode([text])[0].tolist()

    samples, rows, refusals = [], [], []
    for g in golden:
        out = ask(graph, g["question"])
        answerable = bool(g["reference_pages"])
        refused = out["answer"].strip() == NO_ANSWER
        rows.append(
            {
                "question": g["question"],
                "answer": out["answer"],
                "refused": refused,
                "sources": [s["path"] for s in out.get("sources", [])],
            }
        )
        if not answerable:
            refusals.append(refused)
            continue
        samples.append(
            SingleTurnSample(
                user_input=g["question"],
                response=out["answer"],
                reference=g["reference"],
                retrieved_contexts=[s["text"] for s in out.get("sources", [])] or [""],
            )
        )

    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextRecall()],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(IndexEmbeddings()),
        show_progress=False,
    )
    df = result.to_pandas()
    summary = {
        c: float(df[c].mean())
        for c in df.columns
        if c not in {"user_input", "response", "reference", "retrieved_contexts"} and df[c].dtype.kind == "f"
    }
    summary["false_refusals"] = sum(r["refused"] for r, g in zip(rows, golden) if g["reference_pages"])
    summary["refusal_accuracy"] = statistics.mean(refusals) if refusals else None
    summary["judge_model"] = model
    return summary, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=".index")
    ap.add_argument("--golden", default=str(HERE / "golden.jsonl"))
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--llm", action="store_true", help="also run generation + Ragas LLM-judged metrics")
    ap.add_argument("--model", help="Ollama model for the agent and judge (default $WIKIRAG_MODEL)")
    ap.add_argument("--min-hit", type=float, default=0.0, help="exit 1 if hit@k falls below this (CI gate)")
    args = ap.parse_args()

    index = WikiIndex.load(args.index)
    golden = load_golden(Path(args.golden))
    report = {
        "date": dt.datetime.now(dt.UTC).date().isoformat(),
        "embed_model": index.embedder.name,
        "chunks": len(index.chunks),
        "pages": len(index.pages),
    }
    report["retrieval"], report["retrieval_rows"] = retrieval_metrics(index, golden, args.k)
    if args.llm:
        report["generation"], report["generation_rows"] = generation_metrics(index, golden, args.model)

    out = HERE / "results" / f"{report['date']}{'-llm' if args.llm else ''}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_rows")}, indent=2))
    for r in report["retrieval_rows"]:
        if not r["hit"]:
            print(f"MISS: {r['question']}  -> {r['top']}", file=sys.stderr)
    return 1 if report["retrieval"][f"hit@{args.k}"] < args.min_hit else 0


if __name__ == "__main__":
    raise SystemExit(main())
