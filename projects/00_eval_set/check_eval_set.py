"""
check_eval_set.py: test the TEST before trusting it.

For every question we check:
  1. The evidence phrase really exists in its source page (otherwise the question is unanswerable).
  2. How many pages contain that phrase. If it appears on many pages, the question is too
     easy, because almost any result would count as a hit.

Run:  uv run python projects/00_eval_set/check_eval_set.py
"""

from ragkit.corpus import load_docs
from ragkit.eval import is_hit, load_questions

docs = load_docs()  # downloads the FastAPI docs the first time
by_id = {d.doc_id: d for d in docs}
questions = load_questions()

problems = 0
print(f"{'id':4} {'in source?':10} {'#pages':6} source")
for q in questions:
    source = by_id.get(q.source)
    found_in_source = source is not None and is_hit(source.text, q.evidence)
    n_pages = sum(is_hit(d.text, q.evidence) for d in docs)  # True counts as 1
    flag = "" if found_in_source and n_pages <= 2 else "  <-- FIX ME"
    problems += bool(flag)
    print(f"{q.id:4} {str(found_in_source):10} {n_pages:<6} {q.source}{flag}")

print(f"\n{len(questions)} questions, {problems} need attention.")
raise SystemExit(1 if problems else 0)  # non-zero exit lets CI fail on a broken eval set
