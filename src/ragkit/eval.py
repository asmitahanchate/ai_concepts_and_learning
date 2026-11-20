"""
eval.py: the "answer key" and the grader.

How grading works
-----------------
Each test question has an *evidence* phrase: a short piece of text that must
appear in a retrieved chunk for that chunk to count as a hit.

Why evidence phrases instead of "the right document ID"?
Because Project 01 cuts the same pages into chunks in different ways. A page ID
can't tell us whether the *specific chunk* we returned contains the answer, but
an evidence phrase can, and it works no matter how the text was chunked.

Metrics (computed by hand so you see exactly what they mean)
------------------------------------------------------------
* Recall@k: fraction of questions whose evidence shows up anywhere in the top-k results.
* MRR (Mean Reciprocal Rank): average of 1/rank of the first hit (0 if no hit in top-k).
      hit at rank 1 -> 1.0, rank 2 -> 0.5, rank 3 -> 0.33 ...
* Avg context chars: how much text we'd send to an LLM (top-k combined).
  Bigger chunks make hits easier but cost more tokens, so we always report both.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable  # type hint for "a function"
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = REPO_ROOT / "eval" / "questions.jsonl"


@dataclass
class Question:
    id: str
    question: str  # what a user would type
    evidence: str  # text that must appear in a retrieved chunk
    source: str  # the page the evidence comes from (for humans and debugging)


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    """Read one JSON object per line (the 'JSON Lines' format)."""
    with path.open(encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Normalisation: make matching robust to Markdown formatting
# ---------------------------------------------------------------------------
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")  # [text](url) -> text
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase and strip Markdown so 'Uses the `text/event-stream`' matches 'uses the text/event-stream'."""
    text = _LINK.sub(r"\1", text)
    text = text.replace("`", "").replace("**", "").replace("*", "")
    text = _WHITESPACE.sub(" ", text)
    return text.lower().strip()


def is_hit(chunk_text: str, evidence: str) -> bool:
    """A chunk is a hit if the normalised evidence is a substring of the normalised chunk."""
    return normalize(evidence) in normalize(chunk_text)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# A "retriever" is any function: (question_text, k) -> list of result texts, best first.
Retriever = Callable[[str, int], list[str]]


@dataclass
class EvalResult:
    recall: dict[int, float]  # {1: 0.4, 3: 0.6, 5: 0.7}
    mrr: float
    avg_context_chars: float  # average total characters in the top-k results
    misses: list[str]  # question IDs with no hit in top-k (look at these!)
    ranks: dict[str, int | None]  # question ID -> rank of first hit (None if missed)


def evaluate(retrieve: Retriever, questions: list[Question], ks: tuple[int, ...] = (1, 3, 5)) -> EvalResult:
    """Run every question through the retriever and score the results."""
    max_k = max(ks)
    ranks: dict[str, int | None] = {}
    context_sizes: list[int] = []

    for q in questions:
        results = retrieve(q.question, max_k)
        context_sizes.append(sum(len(r) for r in results))
        # Find the rank (1-based) of the FIRST result that contains the evidence.
        ranks[q.id] = next((i + 1 for i, text in enumerate(results) if is_hit(text, q.evidence)), None)

    n = len(questions)
    recall = {k: sum(1 for r in ranks.values() if r is not None and r <= k) / n for k in ks}
    mrr = sum(1 / r for r in ranks.values() if r is not None) / n
    return EvalResult(
        recall=recall,
        mrr=mrr,
        avg_context_chars=sum(context_sizes) / n,
        misses=[qid for qid, r in ranks.items() if r is None],
        ranks=ranks,
    )


def format_row(name: str, result: EvalResult) -> str:
    """One Markdown table row, so results paste straight into a README."""
    recalls = " | ".join(f"{v:.2f}" for v in result.recall.values())
    return f"| {name} | {recalls} | {result.mrr:.2f} | {result.avg_context_chars:,.0f} |"


def table_header(ks: tuple[int, ...] = (1, 3, 5)) -> str:
    cols = " | ".join(f"Recall@{k}" for k in ks)
    seps = "|".join("---:" for _ in ks)
    return f"| Strategy | {cols} | MRR | Avg context chars |\n|---|{seps}|---:|---:|"
