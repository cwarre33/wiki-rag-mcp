import pytest

from wikirag.index import WikiIndex


def test_search_ranks_the_matching_page_first(index):
    hits = index.search("inner product nearest neighbor search", k=2)
    assert hits[0].chunk.path == "tools/faiss.md"
    assert hits[0].score >= hits[-1].score


def test_k_is_clamped_to_corpus_size(index):
    assert len(index.search("anything", k=99)) == len(index.chunks)


def test_private_content_is_not_searchable(index):
    assert all("secret" not in h.chunk.path for h in index.search("private operational details", k=10))
    assert "production-systems/secret.md" not in index.pages


def test_save_load_round_trip(index, embedder, tmp_path):
    index.save(tmp_path / "idx")
    loaded = WikiIndex.load(tmp_path / "idx", embedder)
    assert [c.id for c in loaded.chunks] == [c.id for c in index.chunks]
    assert loaded.search("MBR candidates", k=1)[0].chunk.path == "techniques/mbr-decoding.md"


def test_load_rejects_a_different_embedder(index, tmp_path):
    index.save(tmp_path / "idx")

    class Other:
        name = "other"

    with pytest.raises(ValueError, match="built with"):
        WikiIndex.load(tmp_path / "idx", Other())


def test_empty_corpus_is_an_error(tmp_path, embedder):
    with pytest.raises(ValueError, match="no publishable pages"):
        WikiIndex.build(tmp_path, embedder)
