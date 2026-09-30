"""Embedding index over wiki chunks: sentence-transformers embeddings in a FAISS inner-product index."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from .corpus import Chunk, chunk_pages, load_pages

DEFAULT_EMBED_MODEL = os.environ.get("WIKIRAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        c = self.chunk
        return {
            "id": c.id,
            "path": c.path,
            "title": c.title,
            "heading": c.heading,
            "score": round(self.score, 4),
            "text": c.text,
        }


class WikiIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder, pages: dict[str, str]):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        self.chunks = chunks
        self.embedder = embedder
        self.pages = pages  # path -> full markdown body, for get_page
        self._faiss = faiss.IndexFlatIP(vectors.shape[1])
        self._faiss.add(vectors)
        self._vectors = vectors

    @classmethod
    def build(
        cls, wiki_dir: str | Path, embedder: Embedder, public_only: bool = True, **chunk_kw
    ) -> WikiIndex:
        pages = load_pages(wiki_dir, public_only=public_only)
        chunks = chunk_pages(pages, **chunk_kw)
        if not chunks:
            raise ValueError(f"no publishable pages found under {wiki_dir}")
        vectors = embedder.encode([c.text for c in chunks])
        return cls(chunks, vectors, embedder, {p.path: f"# {p.title}\n\n{p.body}" for p in pages})

    def search(self, query: str, k: int = 5) -> list[Hit]:
        k = max(1, min(k, len(self.chunks)))
        q = self.embedder.encode([query])
        scores, idx = self._faiss.search(q, k)
        return [Hit(self.chunks[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "vectors.npy", self._vectors)
        with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in self.chunks)
        (out / "pages.json").write_text(json.dumps(self.pages, ensure_ascii=False), encoding="utf-8")
        (out / "meta.json").write_text(
            json.dumps(
                {"embed_model": self.embedder.name, "chunks": len(self.chunks), "pages": len(self.pages)},
                indent=2,
            )
        )

    @classmethod
    def load(cls, index_dir: str | Path, embedder: Embedder | None = None) -> WikiIndex:
        d = Path(index_dir)
        meta = json.loads((d / "meta.json").read_text())
        embedder = embedder or SentenceTransformerEmbedder(meta["embed_model"])
        if embedder.name != meta["embed_model"]:
            raise ValueError(f"index built with {meta['embed_model']}, got embedder {embedder.name}")
        chunks = [
            Chunk(**json.loads(line))
            for line in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        pages = json.loads((d / "pages.json").read_text(encoding="utf-8"))
        return cls(chunks, np.load(d / "vectors.npy"), embedder, pages)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the wiki embedding index.")
    ap.add_argument("wiki_dir", help="root of the markdown wiki (e.g. ../cameron-wiki/wiki)")
    ap.add_argument("--out", default=".index", help="output directory")
    ap.add_argument(
        "--include-private", action="store_true", help="index every page, not only visibility: public"
    )
    args = ap.parse_args()
    idx = WikiIndex.build(args.wiki_dir, SentenceTransformerEmbedder(), public_only=not args.include_private)
    idx.save(args.out)
    print(f"indexed {len(idx.pages)} pages as {len(idx.chunks)} chunks -> {args.out}")


if __name__ == "__main__":
    main()
