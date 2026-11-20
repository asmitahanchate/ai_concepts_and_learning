"""
run_baseline.py: score a first, simple retriever so later projects have a number to beat.

Baseline = BM25 keyword search over WHOLE PAGES (no chunking yet).
Whole pages make hits easy (big text = likely contains the evidence) but send an
enormous amount of text to the LLM. Watch the "Avg context chars" column.

Run:  uv run python projects/00_eval_set/run_baseline.py
"""

from ragkit.bm25 import BM25
from ragkit.corpus import load_docs
from ragkit.eval import evaluate, format_row, load_questions, table_header

docs = load_docs()
questions = load_questions()
texts = [d.text for d in docs]

index = BM25(texts)  # "indexing": pre-compute word statistics once


def retrieve(question: str, k: int) -> list[str]:
    """Adapter: BM25 returns (doc_index, score); the grader wants a list of texts."""
    return [texts[i] for i, _score in index.search(question, k)]


result = evaluate(retrieve, questions)
print(table_header())
print(format_row("BM25 over whole pages", result))
print(f"\nMissed ({len(result.misses)}): {', '.join(result.misses) or 'none'}")
