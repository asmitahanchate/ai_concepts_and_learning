# AI Concepts and Learning

Hands-on projects where I rebuild each layer of a production RAG (retrieval-augmented generation) system, measure it, and then think through how it would scale.

Every project follows the same loop: **learn the concept → build it small in Python → measure it → review scalability, reliability and availability.**

| # | Project | What it covers | Status |
|---|---|---|---|
| 00 | [Eval set](projects/00_eval_set/) | Test questions, recall@k, MRR, BM25 baseline | ✅ |
| 01 | [Chunking lab](projects/01_chunking_lab/) | 5 chunking strategies, keyword vs embedding search | ✅ |
| 02 | Embedding shoot-out | Comparing embedding models | ⏳ |
| 03 | Vector DB tour | Chroma, Qdrant, Pinecone, HNSW | ⏳ |

## Setup (Mac)

```bash
brew install uv            # Python project manager
uv sync                    # creates .venv and installs everything from uv.lock
uv run pre-commit install  # turns on the secret scanner for every commit
uv run pytest              # quick check that everything works
```

The corpus is the public [FastAPI documentation](https://github.com/fastapi/fastapi) (MIT license). It's downloaded on the first run, pinned to one commit, and never committed here.

## Layout

```
src/ragkit/     shared building blocks (corpus, grader, BM25, embeddings, chunkers)
eval/           the test set (questions + evidence phrases)
projects/NN_*/  one folder per project, each with its own README
tests/          unit tests
```
