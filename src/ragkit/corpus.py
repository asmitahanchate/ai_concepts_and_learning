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
import subprocess  # runs `curl` to download the docs
import zipfile  # opens .zip archives (built into Python)
from dataclasses import dataclass, field  # quick way to make simple data classes
from pathlib import Path  # nicer, cross-platform file paths

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# GitHub serves any commit as a .zip file at this address. No git needed.
FASTAPI_ZIP_URL = "https://github.com/fastapi/fastapi/archive/{commit}.zip"

# Pin an exact commit so the corpus never changes underneath our eval set.
# If FastAPI edits a page, our evidence phrases could stop matching, so we freeze it.
FASTAPI_COMMIT = "05e6df7e906c3decba377c37548019973f40bf15"

# The repo root is two folders up from this file: src/ragkit/corpus.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Downloaded docs go in data/, which is listed in .gitignore.
DATA_DIR = REPO_ROOT / "data"
ZIP_PATH = DATA_DIR / "fastapi.zip"
DOCS_DIR = DATA_DIR / "fastapi-docs"  # only the English docs get unpacked here

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
# Step 1: download (one zip file, no git)
# ---------------------------------------------------------------------------

def download_fastapi_docs(force: bool = False) -> Path:
    """
    Download FastAPI at the pinned commit as ONE .zip file, then unpack only
    the English docs (docs/en/docs/...). Uses `curl`, which every Mac already has.
    """
    if DOCS_DIR.exists() and any(DOCS_DIR.rglob("*.md")) and not force:
        return DOCS_DIR  # already downloaded, nothing to do

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = FASTAPI_ZIP_URL.format(commit=FASTAPI_COMMIT)
    print(f"Downloading FastAPI docs from {url} ...")
    # -L follows GitHub's redirect, -f fails loudly on HTTP errors, -sS hides the progress bar but shows errors.
    subprocess.run(["curl", "-fsSL", "-o", str(ZIP_PATH), url], check=True)

    # The zip contains one top folder like "fastapi-05e6df.../". We keep only .md files
    # under "<top>/docs/en/docs/" and write them into data/fastapi-docs/ with the same sub-folders.
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for name in zf.namelist():
            _top, _sep, rest = name.partition("/")
            if rest.startswith("docs/en/docs/") and rest.endswith(".md"):
                target = DOCS_DIR / rest.removeprefix("docs/en/docs/")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(name))
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
