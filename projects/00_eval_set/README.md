# 00 · Build the test before the thing

**Idea:** before building a search system, decide how to grade it. This project writes the answer key.

## What's here

| File | Job |
|---|---|
| `../../eval/questions.jsonl` | 30 questions about FastAPI, each with an **evidence phrase** that must appear in a retrieved chunk |
| `check_eval_set.py` | Tests the test: every evidence phrase must exist in its source page, and appear on ≤ 2 pages so questions aren't trivially easy |
| `run_baseline.py` | First score to beat: BM25 keyword search over whole pages |
| `../../src/ragkit/eval.py` | The grader: recall@k, MRR, context size |
| `../../src/ragkit/bm25.py` | BM25 written by hand |

## Run it

```bash
uv run python projects/00_eval_set/check_eval_set.py
uv run python projects/00_eval_set/run_baseline.py
```

## Why evidence phrases instead of "correct page ID"?
Project 01 cuts the same pages into chunks in different ways. A page ID can't tell whether the specific chunk returned holds the answer. An evidence phrase can, no matter how the text was cut.

## Results

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Avg context chars |
|---|---:|---:|---:|---:|---:|
| BM25 over whole pages | 0.73 | 0.97 | 1.00 | 0.86 | 38,514 |

**Takeaway:** whole pages make recall easy, but the top 5 are about 38K characters (~10K tokens) per question. That's expensive, slow, and buries the answer in noise ("lost in the middle"). Chunking (Project 01) is about keeping the recall while shrinking the context.

## Check yourself
1. Recall@5 = 0.60 means what, in plain words?
2. Two systems both have Recall@5 = 0.80. Why might one have a much higher MRR?
3. Why can a test set be wrong, and how does `check_eval_set.py` protect against that?

## Scale review
- **Scalability:** 10,000 questions every night means running evaluation as a parallel batch job.
- **Reliability:** run the eval in CI so a change that breaks retrieval fails before it ships (`check_eval_set.py` already exits non-zero on problems).
- **Limits of this set:** 30 questions is a smoke test, not proof. Real eval sets grow from real user questions and failures.

## Learn more
Search for: "precision and recall explained", "mean reciprocal rank", "BM25 explained", "how to evaluate RAG". Paper: RAGAS (Es et al., 2023).
