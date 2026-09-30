# wiki-rag-mcp

Retrieval-augmented Q&A over a markdown knowledge base. It has three parts:

- a **LangGraph** agent that grades its own retrieval and refuses when it has no grounded answer
- an **MCP server** that exposes the wiki to any MCP client, such as Claude Code, Cursor, or the included client
- an **evaluation harness** with retrieval metrics and **Ragas** LLM-as-a-judge scoring

It indexes the public pages of my engineering wiki (an Obsidian-style vault of production-system notes, ADRs, and Kaggle writeups). It works on any folder of markdown with YAML frontmatter.

```
            ┌──────────────┐   search_wiki / get_page / ask   ┌──────────────────┐
 MCP client │ Claude Code, │ ───────────────────────────────▶ │ wikirag.server   │
 (stdio)    │ Cursor, CLI  │ ◀─────────────────────────────── │ (MCP, stdio)     │
            └──────────────┘                                  └────────┬─────────┘
                                                                       │ ask
                        retrieve ─▶ grade ─(relevant)─▶ generate ─▶ cited answer
                           ▲          │
                           └─ rewrite ◀┘ (none relevant, 1 retry) ─▶ refuse
                                                                       │
                             FAISS inner-product index ◀───────────────┘
                             all-MiniLM-L6-v2 embeddings, 384 chunks / 50 pages
```

## What's in it

| Module | What it does |
|---|---|
| `wikirag/corpus.py` | Loads pages and applies the publication filter: `visibility: public` only, minus redacted sections and security tags. Parses Obsidian frontmatter that isn't valid YAML (`related: [[a]], [[b]]`). Renders wikilinks as words. Chunks by heading with overlapping windows. |
| `wikirag/index.py` | sentence-transformers embeddings in a FAISS `IndexFlatIP` (cosine on normalized vectors). Saves and loads to disk, and refuses to load an index built with a different embedding model. |
| `wikirag/graph.py` | LangGraph state machine: retrieve → LLM relevance grading → generate with numbered citations. On no relevant chunks it does one query rewrite, then refuses. Citations to sources the model wasn't given are stripped. |
| `wikirag/server.py` | MCP server (official Python SDK, `MCPServer`). Tools: `search_wiki`, `get_page`, `ask`. Resource: `wiki://pages`. The LLM is built lazily, so search-only clients never need one running. |
| `wikirag/client.py` | MCP client. Launches the server over stdio, lists its tools, and runs a search. |
| `eval/run_eval.py` | Scores the pipeline against `eval/golden.jsonl`. Hand-written questions with reference answers, pages and passages, plus one question the wiki can't answer. |

## Quickstart

```bash
uv venv && uv pip install -e ".[eval,dev]"
wikirag-index ../cameron-wiki/wiki --out .index          # build the index
python -m wikirag.client "why run CLIP as a subprocess"   # MCP client → server → FAISS

ollama pull llama3.2:3b                                  # local LLM for the agent and judge
wikirag-ask "Why does SofaScope use metadata scoring instead of embeddings?"
```

Use it from Claude Code:

```bash
claude mcp add wiki -- /path/to/.venv/bin/python -m wikirag.server --index /path/to/.index
```

## Evaluation

```bash
python eval/run_eval.py --index .index          # retrieval metrics (no LLM)
python eval/run_eval.py --index .index --llm    # + Ragas faithfulness, answer relevancy, context recall
```

Retrieval baseline, 15 answerable questions, k = 5 (`eval/results/2026-09-30.json`):

| Metric | Score |
|---|---|
| hit@5 (a reference page in the top 5) | 0.93 |
| MRR | 0.86 |
| Snippet recall (reference passages present verbatim in retrieved text) | 0.57 |

The one miss asks about the trade-offs of the SofaScope metadata-scoring ADR. It retrieves the system overview and the hybrid-routing page instead, which discuss the same decision. Snippet recall is lower than hit rate because heading-based windows split some reference passages across chunks. That is the next thing to tune.

The `--llm` run evaluates generation with Ragas and uses the local Ollama model as both agent and judge:
- `Faithfulness`
- `ResponseRelevancy`
- `LLMContextRecall`
- refusal accuracy on the unanswerable question

Generation, 2026-09-30, 15 answerable questions plus 1 unanswerable. `qwen2.5:7b` is both the agent and the Ragas judge, running locally on Ollama:

| Metric | qwen2.5:7b | llama3.2:3b (first run) |
|---|---|---|
| Faithfulness | 0.73 | 0.50 |
| LLM context recall | 0.81 | 0.80 |
| Answer relevancy | 0.52 ⚠️ | ~1.00 ⚠️ |
| False refusals (answerable questions refused) | 0 | — |
| Judge exceptions | 0 | 25 |

**Why the 3B run isn't trustworthy.** It threw 25 judge exceptions, and Ragas records a failed judgment as NaN. Pandas' `mean()` silently skips NaN, so each 3B average may cover only a few of the 15 questions. That is the most likely reason answer relevancy came out at a suspiciously perfect ~1.00.

The eval now:
- records how many samples each metric actually scored (`<metric>_scored`)
- saves per-question scores to the results file
- writes one results file per judge model instead of overwriting

**Answer relevancy is under investigation.** Ragas `ResponseRelevancy` asks the judge to generate questions from the answer. It scores the cosine similarity between those questions and the original, using the MiniLM embedder, and scores 0 when the judge flags the answer as noncommittal. Short, cited answers and a small embedder both pull the score down. The per-question scores from the next run will show which is driving it.

## Tests

```bash
pytest -q        # 25 tests; deterministic hash embedder + fake chat model, no downloads
ruff check .
```

The tests cover the publication filter (private and security-tagged pages never reach the index or the MCP surface), chunking, index persistence, and every graph path: answer, rewrite then recover, rewrite then refuse. They also cover citation stripping, and the MCP tools and resource over an in-process client.

## Design notes

- **Grading before generating.** Small local models are easily distracted by off-topic context, so only chunks the grader keeps reach the answer prompt. (Measuring this with `--llm` runs, with and without grading, is on the to-do list.) Grading also gives the refusal path a clean trigger.
- **The publication filter lives at the index boundary.** The server only ever sees what was indexed, so there is no per-request access check to get wrong.
- **One rewrite, not a loop.** It keeps latency bounded and makes refusal deterministic on questions the wiki can't answer.

## License

MIT
