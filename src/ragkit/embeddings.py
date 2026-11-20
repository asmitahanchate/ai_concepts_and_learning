"""
embeddings.py: turn text into vectors ("meaning fingerprints"), with a cache.

An embedding model maps text -> a list of numbers (a vector). Texts with similar
meaning land close together, so "How do I upload a file?" ends up near the
paragraph about UploadFile even though they share few exact words.

We use BAAI/bge-small-en-v1.5: free, small (~130 MB), runs on a laptop CPU.
It downloads from Hugging Face the first time you run it.

Cache: embedding is the slowest step, so we store each vector in a small SQLite
database keyed by a hash of (model name + text). Re-running only embeds text that
changed. Production systems do the same thing at scale ("don't re-embed unchanged content").
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / ".cache" / "embeddings.sqlite"  # .cache/ is gitignored


class EmbeddingCache:
    """Tiny persistent dict: sha256(model + text) -> float32 vector bytes."""

    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS vec (key TEXT PRIMARY KEY, data BLOB)")

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(keys), 500):  # SQLite limits how many "?" one query may have
            batch = keys[start : start + 500]
            marks = ",".join("?" * len(batch))
            for k, blob in self.db.execute(f"SELECT key, data FROM vec WHERE key IN ({marks})", batch):
                found[k] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO vec VALUES (?, ?)",
            [(k, v.astype(np.float32).tobytes()) for k, v in items.items()],
        )
        self.db.commit()


class Embedder:
    """Base class: subclasses implement _embed(); this class adds caching and normalisation."""

    name = "base"
    query_prefix = ""  # some models expect an instruction in front of search queries

    def __init__(self, use_cache: bool = True):
        self.cache = EmbeddingCache() if use_cache else None

    def _embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed many texts -> array of shape (len(texts), dim). Each row has length 1."""
        keys = [EmbeddingCache.key(self.name, t) for t in texts]
        cached = self.cache.get_many(keys) if self.cache else {}
        todo = [i for i, k in enumerate(keys) if k not in cached]
        if todo:
            fresh = self._embed([texts[i] for i in todo])
            new_items = {keys[i]: vec for i, vec in zip(todo, fresh)}
            cached.update(new_items)
            if self.cache:
                self.cache.put_many(new_items)
        vectors = np.stack([cached[k] for k in keys])
        # Normalise to length 1, so cosine similarity becomes a simple dot product.
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([self.query_prefix + text])[0]


class BGESmallEmbedder(Embedder):
    """The real thing: a small open-source embedding model from Hugging Face."""

    name = "BAAI/bge-small-en-v1.5"
    # BGE v1.5 recommends this instruction for queries (not for documents).
    query_prefix = "Represent this sentence for searching relevant passages: "

    def __init__(self, use_cache: bool = True):
        super().__init__(use_cache)
        from sentence_transformers import SentenceTransformer  # imported here: it's slow to load

        self.model = SentenceTransformer(self.name)

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, show_progress_bar=len(texts) > 200)


class HashingEmbedder(Embedder):
    """
    OFFLINE STAND-IN, not a real semantic model.
    Hashes words and word-pairs into a fixed-size vector. It only captures word overlap,
    so it is useful for tests and for machines that can't download models, but its
    scores must not be reported as "embedding" results.
    """

    name = "hashing-offline-standin"

    def __init__(self, dim: int = 1024, use_cache: bool = False):
        super().__init__(use_cache)
        self.dim = dim

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = re.findall(r"[a-z0-9_]+", text.lower())
            for feature in words + [f"{a} {b}" for a, b in zip(words, words[1:])]:
                h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        return out


def get_embedder() -> Embedder:
    """Pick the embedder from the RAGKIT_EMBEDDER env var (default: the real BGE model)."""
    choice = os.getenv("RAGKIT_EMBEDDER", "bge-small")
    return HashingEmbedder() if choice == "hashing" else BGESmallEmbedder()
