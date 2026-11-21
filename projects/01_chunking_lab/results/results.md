# Chunking lab results

Corpus: 114 FastAPI doc pages. Eval set: 30 questions.

## Chunk statistics

| Strategy | Chunks | Avg chars | Max chars |
|---|---:|---:|---:|
| 1. Fixed 500 | 1,301 | 477 | 500 |
| 2. Recursive 500 | 1,570 | 402 | 499 |
| 3. Markdown headers | 1,049 | 650 | 1,643 |
| 4. Semantic | 1,084 | 566 | 1,524 |
| 5. Small-to-big | 4,584 | 136 | 200 |

## BM25 (keyword) retrieval

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Avg context chars |
|---|---:|---:|---:|---:|---:|
| 1. Fixed 500 | 0.40 | 0.57 | 0.67 | 0.50 | 2,407 |
| 2. Recursive 500 | 0.33 | 0.57 | 0.63 | 0.45 | 2,038 |
| 3. Markdown headers | 0.47 | 0.67 | 0.77 | 0.56 | 3,776 |
| 4. Semantic | 0.47 | 0.73 | 0.73 | 0.59 | 4,421 |
| 5. Small-to-big | 0.30 | 0.53 | 0.63 | 0.44 | 4,435 |

## Dense retrieval: BAAI/bge-small-en-v1.5

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Avg context chars |
|---|---:|---:|---:|---:|---:|
| 1. Fixed 500 | 0.30 | 0.67 | 0.77 | 0.48 | 2,463 |
| 2. Recursive 500 | 0.47 | 0.67 | 0.73 | 0.57 | 2,077 |
| 3. Markdown headers | 0.37 | 0.63 | 0.80 | 0.54 | 3,595 |
| 4. Semantic | 0.37 | 0.67 | 0.77 | 0.51 | 4,255 |
| 5. Small-to-big | 0.43 | 0.63 | 0.73 | 0.55 | 4,340 |
