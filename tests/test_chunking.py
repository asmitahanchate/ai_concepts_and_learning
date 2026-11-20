from ragkit import chunking
from ragkit.corpus import Doc
from ragkit.embeddings import HashingEmbedder

DOC = Doc("d.md", "Title", "# Title\n\nIntro text here.\n\n## Part A\n\n" + "Alpha sentence. " * 60
          + "\n\n## Part B\n\n" + "Beta sentence. " * 60)


def test_fixed_size_respects_limit():
    assert all(len(c.search_text) <= 500 for c in chunking.fixed_size([DOC]))


def test_markdown_chunks_carry_breadcrumbs():
    chunks = chunking.markdown_headers([DOC])
    assert any(c.search_text.startswith("[Title > Part B]") for c in chunks)


def test_small_to_big_returns_parent():
    chunks = chunking.small_to_big([DOC])
    assert all(len(c.search_text) <= 200 for c in chunks)  # children are small
    assert all(c.search_text in c.context_text for c in chunks)  # child lives inside its parent


def test_semantic_runs_and_respects_max_size():
    chunks = chunking.semantic([DOC], HashingEmbedder(), max_size=400)
    assert chunks and all(len(c.search_text) <= 400 + 20 for c in chunks)
