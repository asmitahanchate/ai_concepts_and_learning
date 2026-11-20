"""
corpus.py: download and load the documents we search over.

The corpus is the public FastAPI documentation (MIT licensed), written in Markdown.
We never commit the docs to this repo. Instead we download them on demand,
pinned to one exact Git commit, so everyone running this gets identical text
and therefore comparable scores.

Big idea: a RAG system is only as good as the text you feed it. This file is
the "textbook" in the open-book-exam analogy.
"""

from __future__ import annotations  # lets us use modern type hints on older Pythons

import re  # regular expressions, used to clean Markdown
import subprocess  # runs `git` commands from Python
from dataclasses import dataclass, field  # quick way to make simple data classes
from pathlib import Path  # nicer, cross-platform file paths

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Where the repo lives on GitHub.
FASTAPI_REPO = "https://github.com/fastapi/fastapi.git"

# Pin an exact commit so the corpus never changes underneath our eval set.
# If FastAPI edits a page, our evidence phrases could stop matching, so we freeze it.
FASTAPI_COMMIT = "05e6df7e906c3decba377c37548019973f40bf15"

# The repo root is two folders up from this file: src/ragkit/corpus.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Downloaded docs go in data/, which is listed in .gitignore.
DATA_DIR = REPO_ROOT / "data"
CLONE_DIR = DATA_DIR / "fastapi-src"
DOCS_DIR = CLONE_DIR / "docs" / "en" / "docs"

# Sections of the docs we include. These are the "learning" pages.
INCLUDE_DIRS = ["tutorial", "advanced", "deployment", "how-to"]

# Individual top-level pages that explain concepts (async, types, and so on).
INCLUDE_TOP_LEVEL = [
    "async.md",
    "python-types.md",
    "features.md",
    "fastapi-cli.md",
    "virtual-environments.md",
    "environment-variables.md",
]
# Deliberately excluded: release-notes.md (huge changelog), reference/ (auto-generated
# API listings), and community pages (people, newsletter). They add noise, not knowledge.


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class Doc:
    """One documentation page."""

    doc_id: str  # stable ID: the file path relative to the docs folder, e.g. "tutorial/path-params.md"
    title: str  # the page's first heading
    text: str  # cleaned Markdown text
    metadata: dict = field(default_factory=dict)  # anything extra (section, file size, ...)


# ---------------------------------------------------------------------------
# Step 1: download
# ---------------------------------------------------------------------------

def _git(*args: str, cwd: Path | None = None) -> None:
    """Run a git command and stop with a clear error if it fails."""
    subprocess.run(["git", *args], cwd=cwd, check=True)


def download_fastapi_docs(force: bool = False) -> Path:
    """
    Fetch ONLY the docs folder at the pinned commit.

    Uses two Git features so we don't download the whole repo:
      * --filter=blob:none  -> "partial clone": file contents are fetched lazily
      * sparse-checkout     -> only check out docs/en/docs
    """
    if DOCS_DIR.exists() and not force:
        return DOCS_DIR  # already downloaded, nothing to do

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CLONE_DIR.exists():
        _git("clone", "--filter=blob:none", "--no-checkout", "--quiet", FASTAPI_REPO, str(CLONE_DIR))
    _git("sparse-checkout", "set", "docs/en/docs", cwd=CLONE_DIR)
    # Fetch exactly our pinned commit (shallow = no history) and check it out.
    _git("fetch", "--depth", "1", "--quiet", "origin", FASTAPI_COMMIT, cwd=CLONE_DIR)
    _git("checkout", "--quiet", FASTAPI_COMMIT, cwd=CLONE_DIR)
    return DOCS_DIR


# ---------------------------------------------------------------------------
# Step 2: clean
# ---------------------------------------------------------------------------

# FastAPI's docs use a few special Markdown extensions. Each pattern below
# removes one of them so the text reads like plain prose.
_HEADING_ANCHOR = re.compile(r"\s*\{\s*#[\w\-]+\s*\}")  # "## Title { #title }" -> "## Title"
_CODE_INCLUDE = re.compile(r"^\{\*.*?\*\}\s*$", re.MULTILINE)  # "{* ../../docs_src/x.py *}" lines
_DFN_TAG = re.compile(r"<dfn[^>]*>(.*?)</dfn>", re.DOTALL)  # "<dfn title='...'>word</dfn>" -> "word"
_ABBR_TAG = re.compile(r"<abbr[^>]*>(.*?)</abbr>", re.DOTALL)  # same idea for <abbr>
_ADMONITION = re.compile(r"^///.*$", re.MULTILINE)  # "/// tip" ... "///" box markers
_MANY_BLANK_LINES = re.compile(r"\n{3,}")  # squash 3+ newlines down to 2


def clean_markdown(text: str) -> str:
    """Turn FastAPI-flavoured Markdown into clean Markdown."""
    text = _HEADING_ANCHOR.sub("", text)
    # Code examples live in separate .py files we didn't download; leave a marker
    # so a reader (or an LLM) knows a code sample was here.
    text = _CODE_INCLUDE.sub("[code example]", text)
    text = _DFN_TAG.sub(r"\1", text)
    text = _ABBR_TAG.sub(r"\1", text)
    text = _ADMONITION.sub("", text)
    text = _MANY_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _first_heading(text: str, fallback: str) -> str:
    """Use the first '# Heading' line as the page title."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


# ---------------------------------------------------------------------------
# Step 3: load
# ---------------------------------------------------------------------------

def _selected_files(docs_dir: Path) -> list[Path]:
    """List the Markdown files we want, in a stable (sorted) order."""
    files: list[Path] = []
    for folder in INCLUDE_DIRS:
        files.extend(sorted((docs_dir / folder).rglob("*.md")))
    for name in INCLUDE_TOP_LEVEL:
        path = docs_dir / name
        if path.exists():
            files.append(path)
    return files


def load_docs(download: bool = True) -> list[Doc]:
    """Download (if needed), clean, and return every page as a Doc."""
    docs_dir = download_fastapi_docs() if download else DOCS_DIR
    docs: list[Doc] = []
    for path in _selected_files(docs_dir):
        raw = path.read_text(encoding="utf-8")
        text = clean_markdown(raw)
        if len(text) < 200:  # skip near-empty stub pages
            continue
        doc_id = path.relative_to(docs_dir).as_posix()
        docs.append(
            Doc(
                doc_id=doc_id,
                title=_first_heading(text, fallback=doc_id),
                text=text,
                metadata={"section": doc_id.split("/")[0], "chars": len(text)},
            )
        )
    return docs


if __name__ == "__main__":
    # Quick sanity check:  uv run python -m ragkit.corpus
    pages = load_docs()
    total = sum(len(d.text) for d in pages)
    print(f"Loaded {len(pages)} pages, {total:,} characters")
    for d in pages[:5]:
        print(f"  {d.doc_id:45s} {d.title}")
