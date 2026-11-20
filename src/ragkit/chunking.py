"""
chunking.py: five ways to cut documents into searchable pieces.

Every strategy returns a list of Chunk objects with two text fields:
  * search_text:  what we index (embed / BM25)
  * context_text: what we hand back to the LLM
For four strategies they're identical. For small-to-big they differ: we SEARCH
small, precise children but RETURN their bigger parent section.

Strategies 1-3 use LangChain's text splitters (the industry-standard library).
Strategies 4-5 are written by hand so you can see exactly how they work.
LangChain equivalents: SemanticChunker (langchain_experimental) and ParentDocumentRetriever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ragkit.corpus import Doc
from ragkit.embeddings import Embedder


@dataclass
class Chunk:
    search_text: str
    context_text: str
    doc_id: str
    metadata: dict = field(default_factory=dict)


def _same(text: str, doc: Doc, **meta) -> Chunk:
    """Helper for strategies where we search and return the same text."""
    return Chunk(search_text=text, context_text=text, doc_id=doc.doc_id, metadata=meta)


# 1. FIXED SIZE -------------------------------------------------------------
def fixed_size(docs: list[Doc], size: int = 500) -> list[Chunk]:
    """Cut every `size` characters, ignoring sentences. Simple, but splits ideas in half."""
    splitter = CharacterTextSplitter(separator="", chunk_size=size, chunk_overlap=0)
    return [_same(t, d) for d in docs for t in splitter.split_text(d.text)]


# 2. RECURSIVE ---------------------------------------------------------------
def recursive(docs: list[Doc], size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Try to split on paragraphs, then lines, then sentences, then words, so pieces stay whole.
    `overlap` repeats a little text between neighbours so an idea cut at a border survives."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return [_same(t, d) for d in docs for t in splitter.split_text(d.text)]


# 3. MARKDOWN HEADERS (document-aware) ---------------------------------------
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _sections(doc: Doc) -> list[tuple[str, str]]:
    """Split one page by its headings. Returns (breadcrumb, section_text) pairs."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=False)
    out = []
    for part in splitter.split_text(doc.text):
        crumb = " > ".join(part.metadata[h] for _, h in _HEADERS if h in part.metadata)
        out.append((crumb, part.page_content))
    return out


def markdown_headers(docs: list[Doc], max_size: int = 1500) -> list[Chunk]:
    """One chunk per section. Long sections are split again (recursively), and every
    piece gets its heading 'breadcrumb' in front, e.g. 'Request Files > File Parameters',
    so a chunk still makes sense on its own."""
    sub = RecursiveCharacterTextSplitter(chunk_size=max_size, chunk_overlap=100)
    chunks = []
    for d in docs:
        for crumb, text in _sections(d):
            for piece in sub.split_text(text) if len(text) > max_size else [text]:
                chunks.append(_same(f"[{crumb}]\n{piece}", d, section=crumb))
    return chunks


# 4. SEMANTIC ------------------------------------------------------------------
_SENTENCE_END = re.compile(r"(?<=[.!?:])\s+|\n{2,}")


def semantic(docs: list[Doc], embedder: Embedder, percentile: float = 90, max_size: int = 1500) -> list[Chunk]:
    """Cut where the TOPIC changes.
    1. Split a page into sentences.  2. Embed each sentence.
    3. Measure the distance (1 - cosine similarity) between neighbouring sentences.
    4. Cut wherever that distance is unusually large (above the page's Nth percentile),
       or when a chunk would grow past max_size."""
    chunks = []
    for d in docs:
        sentences = [s.strip() for s in _SENTENCE_END.split(d.text) if s and s.strip()]
        if len(sentences) < 3:
            chunks.append(_same(d.text, d))
            continue
        vecs = embedder.embed_documents(sentences)
        distances = 1 - np.sum(vecs[:-1] * vecs[1:], axis=1)  # distance between sentence i and i+1
        threshold = np.percentile(distances, percentile)
        current = [sentences[0]]
        for sentence, dist in zip(sentences[1:], distances):
            too_big = sum(len(s) for s in current) + len(sentence) > max_size
            if dist > threshold or too_big:
                chunks.append(_same(" ".join(current), d))
                current = []
            current.append(sentence)
        chunks.append(_same(" ".join(current), d))
    return chunks


# 5. SMALL-TO-BIG (parent / child) ----------------------------------------------
def small_to_big(docs: list[Doc], child_size: int = 200, parent_max: int = 2000) -> list[Chunk]:
    """Parents = heading sections (capped at parent_max chars).
    Children = small ~200-char pieces of each parent.
    We index the children (precise matching) but return the parent (full context)."""
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_max, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=20)
    chunks = []
    for d in docs:
        for crumb, section in _sections(d):
            for p_idx, parent in enumerate(parent_splitter.split_text(section)):
                parent_text = f"[{crumb}]\n{parent}"
                parent_id = f"{d.doc_id}#{crumb}#{p_idx}"
                for child in child_splitter.split_text(parent):
                    chunks.append(Chunk(child, parent_text, d.doc_id, {"parent_id": parent_id}))
    return chunks
