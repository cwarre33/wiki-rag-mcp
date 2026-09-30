import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from run_eval import normalize, retrieval_metrics


def test_retrieval_metrics(index):
    golden = [
        {
            "question": "FAISS inner product search",
            "reference_pages": ["tools/faiss.md"],
            "reference_contexts": ["FAISS **IndexFlatIP** performs exact inner product"],
        },
        {"question": "pizza", "reference_pages": [], "reference_contexts": []},
    ]
    summary, rows = retrieval_metrics(index, golden, k=1)
    assert summary == {"hit@1": 1, "mrr": 1.0, "snippet_recall": 1.0, "questions": 1}
    assert rows[0]["top"] == ["tools/faiss.md"]


def test_normalize_strips_markdown():
    assert normalize("**Bold**  `code`\n# H") == "bold code h"
