"""
run_lab.py: chunk the FastAPI docs 5 ways and grade each with the Project 00 eval set.

For every strategy we measure:
  * chunk stats: how many chunks, average and max size
  * retrieval quality with TWO searchers:
      - BM25 (keyword) over the chunks
      - dense embeddings (BGE-small) over the chunks
  * avg context chars: how much text the top-5 would cost us in an LLM prompt

Run:   uv run python projects/01_chunking_lab/run_lab.py
Output: results/results.md and results/recall_vs_context.png
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw charts to a file, no window needed
import matplotlib.pyplot as plt

from ragkit import chunking
from ragkit.bm25 import BM25
from ragkit.chunking import Chunk
from ragkit.corpus import load_docs
from ragkit.embeddings import HashingEmbedder, get_embedder
from ragkit.eval import EvalResult, evaluate, format_row, load_questions, table_header
from ragkit.vector_search import VectorIndex

OUT_DIR = Path(__file__).parent / "results"


def dedupe_contexts(chunks: list[Chunk], hits: list[int], k: int) -> list[str]:
    """Turn ranked chunk indices into up to k UNIQUE context texts.
    Matters for small-to-big: three children of the same parent should count once."""
    seen, out = set(), []
    for i in hits:
        ctx = chunks[i].context_text
        if ctx not in seen:
            seen.add(ctx)
            out.append(ctx)
        if len(out) == k:
            break
    return out


def bm25_retriever(chunks: list[Chunk]):
    index = BM25([c.search_text for c in chunks])
    # Ask for extra candidates (k*4) so de-duplication still leaves k results.
    return lambda q, k: dedupe_contexts(chunks, [i for i, _ in index.search(q, k * 4)], k)


def dense_retriever(chunks: list[Chunk], embedder):
    index = VectorIndex(embedder.embed_documents([c.search_text for c in chunks]))
    return lambda q, k: dedupe_contexts(chunks, [i for i, _ in index.search(embedder.embed_query(q), k * 4)], k)


def main() -> None:
    docs = load_docs()
    questions = load_questions()
    embedder = get_embedder()
    offline = isinstance(embedder, HashingEmbedder)
    dense_label = "hashing stand-in (NOT semantic)" if offline else embedder.name
    print(f"{len(docs)} pages, {len(questions)} questions, dense searcher: {dense_label}\n")

    strategies = {
        "1. Fixed 500": lambda: chunking.fixed_size(docs),
        "2. Recursive 500": lambda: chunking.recursive(docs),
        "3. Markdown headers": lambda: chunking.markdown_headers(docs),
        "4. Semantic": lambda: chunking.semantic(docs, embedder),
        "5. Small-to-big": lambda: chunking.small_to_big(docs),
    }

    stats_rows, bm25_rows, dense_rows = [], [], []
    points: dict[str, tuple[EvalResult, EvalResult]] = {}
    for name, make in strategies.items():
        t0 = time.time()
        chunks = make()
        sizes = [len(c.search_text) for c in chunks]
        stats_rows.append(f"| {name} | {len(chunks):,} | {sum(sizes) / len(sizes):,.0f} | {max(sizes):,} |")
        bm = evaluate(bm25_retriever(chunks), questions)
        dn = evaluate(dense_retriever(chunks, embedder), questions)
        bm25_rows.append(format_row(name, bm))
        dense_rows.append(format_row(name, dn))
        points[name] = (bm, dn)
        print(f"{name:22s} {len(chunks):5,} chunks  BM25 R@5={bm.recall[5]:.2f}  dense R@5={dn.recall[5]:.2f}  ({time.time() - t0:.1f}s)")

    OUT_DIR.mkdir(exist_ok=True)
    report = "\n".join([
        "# Chunking lab results", "",
        f"Corpus: {len(docs)} FastAPI doc pages. Eval set: {len(questions)} questions.", "",
        "## Chunk statistics", "", "| Strategy | Chunks | Avg chars | Max chars |", "|---|---:|---:|---:|", *stats_rows, "",
        "## BM25 (keyword) retrieval", "", table_header(), *bm25_rows, "",
        f"## Dense retrieval: {dense_label}", "", table_header(), *dense_rows, "",
    ])
    (OUT_DIR / "results.md").write_text(report)

    # Chart: quality (Recall@5) vs cost (context chars). Up-and-left is best.
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (bm, dn) in points.items():
        for res, marker, label in ((bm, "o", "BM25"), (dn, "s", "dense")):
            ax.scatter(res.avg_context_chars, res.recall[5], marker=marker, s=70)
            ax.annotate(f"{name.split('. ')[1]} ({label})", (res.avg_context_chars, res.recall[5]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Avg context chars sent to the LLM (top-5)  -> more cost")
    ax.set_ylabel("Recall@5  -> better")
    ax.set_title("Chunking strategies: quality vs context cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recall_vs_context.png", dpi=150)
    print(f"\nWrote {OUT_DIR / 'results.md'} and recall_vs_context.png")


if __name__ == "__main__":
    main()
