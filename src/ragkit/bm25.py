"""
bm25.py: classic keyword search, written by hand.

BM25 is the scoring formula behind most keyword search engines (Elasticsearch,
OpenSearch, Lucene). It is the baseline every fancy embedding search must beat.

The intuition, in three rules:
  1. A document that mentions a query word MORE often is more relevant...
  2. ...but with diminishing returns (the 10th mention adds less than the 2nd)   <- k1
  3. Rare words matter more than common words ("threadpool" > "the")             <- IDF
  4. Long documents naturally contain more words, so we normalise by length      <- b

Score(doc, query) = sum over query words w of:
        IDF(w) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Very common English words that carry no meaning for search.
STOPWORDS = set(
    "a an and are as at be by can do does for from how i if in is it my of on or "
    "so that the this to what when which why with you your".split()
)
_TOKEN = re.compile(r"[a-z0-9_]+")  # words, numbers and snake_case identifiers


def tokenize(text: str) -> list[str]:
    """Lowercase, split into word tokens, drop stopwords."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """An in-memory BM25 index over a list of texts."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.texts = texts
        self.k1 = k1  # how quickly repeated words stop adding score
        self.b = b  # how much to penalise long documents (0 = not at all, 1 = fully)

        # Pre-compute everything we can at index time (this is what "indexing" means).
        self.doc_tokens = [tokenize(t) for t in texts]
        self.doc_freqs = [Counter(toks) for toks in self.doc_tokens]  # word -> count, per doc
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avg_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)

        # Document frequency: in how many docs does each word appear?
        df: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        n = len(texts)
        # IDF: rare words get a high weight. The +1 inside log keeps it positive.
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def score(self, query: str, i: int) -> float:
        """BM25 score of document i for the query."""
        freqs, length = self.doc_freqs[i], self.doc_lens[i]
        total = 0.0
        for w in tokenize(query):
            tf = freqs.get(w, 0)
            if tf == 0:
                continue  # word not in this doc: contributes nothing
            norm = 1 - self.b + self.b * length / self.avg_len
            total += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return total

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return the top-k (doc_index, score) pairs, best first."""
        scores = [(i, self.score(query, i)) for i in range(len(self.texts))]
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return [pair for pair in scores[:k] if pair[1] > 0]
