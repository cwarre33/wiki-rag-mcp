from wikirag.corpus import Page, chunk_page, clean_links, load_pages, parse_frontmatter


def test_only_public_unredacted_pages_load(wiki_dir):
    paths = {p.path for p in load_pages(wiki_dir)}
    assert paths == {"techniques/mbr-decoding.md", "tools/faiss.md"}


def test_include_private_loads_everything(wiki_dir):
    assert len(load_pages(wiki_dir, public_only=False)) == 5


def test_obsidian_frontmatter_that_is_not_yaml_still_parses():
    fm, body = parse_frontmatter(
        "---\ntitle: X\nvisibility: public\nrelated: [[a.md]], [[b.md]]\ntags: [one, two]\n---\nbody"
    )
    assert fm["visibility"] == "public"
    assert fm["tags"] == ["one", "two"]
    assert body == "body"


def test_wikilinks_render_as_words():
    assert (
        clean_links("see [[wiki/models/byt5.md|ByT5]] and [[wiki/tools/faiss-gpu.md]]")
        == "see ByT5 and faiss gpu"
    )


def test_chunks_carry_title_and_heading():
    page = Page(
        "t/p.md", "Title", "t", (), "# Title\n\n## Part A\n" + "alpha " * 20 + "\n## Part B\n" + "beta " * 20
    )
    chunks = chunk_page(page)
    assert [c.heading for c in chunks] == ["Part A", "Part B"]
    assert all(c.text.startswith("Title — ") for c in chunks)
    assert chunks[0].id == "t/p.md#0"


def test_long_sections_are_windowed_with_overlap():
    body = "## Long\n" + "\n".join(f"line {i} " + "x" * 60 for i in range(60))
    chunks = chunk_page(Page("p.md", "P", "", (), body), max_chars=500, overlap=100)
    assert len(chunks) > 5
    assert all(len(c.text) <= 500 + len("P — Long\n") for c in chunks)
