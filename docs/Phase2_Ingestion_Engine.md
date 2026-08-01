# Phase 2 — The Ingestion Engine (Pipeline 1)

> **Prerequisite:** Phase 1 complete. This phase imports `Document`, `Section`, `Chunk`,
> `ChunkLevel`, the `Base*` interfaces, `settings`, `logger`, and every helper in
> `src/core/utils.py`.
>
> **Budget:** ~1,600 lines of Python across 11 files. (Budgeted at 1,000 in the roadmap; see the
> calibration note there — scale machinery is bulkier than it looks.)
> **Depends on:** Phase 1 only. Does **not** depend on Phase 3 — this phase produces `Chunk` objects
> and stops. Storing them is Phase 3's job.

---

## 1. What Makes This Phase Hard

Ingestion sounds like the boring part: read files, cut them up, done. At 500 documents it is. At
**650,833 documents and 37 GB**, four things break that do not break at small scale, and every design
decision in this phase exists because of one of them.

**You cannot hold the corpus in memory.** 37 GB does not fit in RAM. Any code shaped like
`documents = load_all(); process(documents)` is dead on arrival. Everything must stream.

**The job takes hours, so it will be interrupted.** A run over 650k documents takes many hours. Your
laptop will sleep, or you will hit Ctrl+C, or a single malformed file will raise at document 480,000.
If the only recovery is "start again", the pipeline is unusable. It must checkpoint and resume.

**Thousands of documents will be broken.** At 650k files, even a 0.5% failure rate is 3,250 files with
mangled encodings, zero bytes, binary content mislabelled `.txt`, or structure no parser can make
sense of. A pipeline that crashes on bad input will never finish. It must quarantine and continue.

**Embedding is CPU-bound, and `asyncio` cannot help.** This is the counter-intuitive one. Everywhere
else in this project, `async` is the answer to slowness. Here it is useless: generating embeddings is
raw computation, and `asyncio` gives you concurrency, not parallelism. On a single core, 650k
embeddings take as long as they take. Escaping that requires `multiprocessing`.

So this phase is really two things braided together: a **document-processing pipeline** (load → parse →
chunk), which is where the RAG-specific thinking lives, and a **scale harness** (streaming,
checkpointing, quarantine, parallelism), which is where the engineering lives.

### The files

| # | File | Lines | Role |
| :-- | :--- | ---: | :--- |
| | **Part A — document processing** | | |
| 1 | `src/ingestion/loaders/base.py` | 60 | Shared loader behaviour |
| 2 | `src/ingestion/loaders/txt_loader.py` | 130 | Encoding-robust text loading |
| 3 | `src/ingestion/loaders/factory.py` | 60 | Dispatch by file extension |
| 4 | `src/ingestion/parsers/legal_parser.py` | 220 | `SECTION` / `ARTICLE` segmentation |
| 5 | `src/ingestion/parsers/metadata_extractor.py` | 140 | Contract name, year, parties |
| 6 | `src/ingestion/chunkers/sentence_chunker.py` | 230 | Sentence-aligned token chunking |
| 7 | `src/ingestion/chunkers/parent_child_chunker.py` | 150 | Small-to-search, large-to-generate |
| | **Part B — scale harness** | | |
| 8 | `src/ingestion/discovery.py` | 130 | Generator walk over the corpus |
| 9 | `src/ingestion/checkpoint.py` | 180 | Resumable state |
| 10 | `src/ingestion/dead_letter.py` | 130 | Quarantine for failures |
| 11 | `src/ingestion/pipeline.py` | 280 | Multiprocessing orchestrator |

### Directory to create

```text
src/ingestion/
├── __init__.py
├── discovery.py
├── checkpoint.py
├── dead_letter.py
├── pipeline.py
├── loaders/
│   ├── __init__.py
│   ├── base.py
│   ├── txt_loader.py
│   └── factory.py
├── parsers/
│   ├── __init__.py
│   ├── legal_parser.py
│   └── metadata_extractor.py
└── chunkers/
    ├── __init__.py
    ├── sentence_chunker.py
    └── parent_child_chunker.py
```

Also rename `notesbooks/` → `notebooks/` while you are here, to match the roadmap's tree. Trivial, but
it will otherwise annoy you for fifteen more phases.

### Data flow

```text
discovery.py            walks data/contracts/<year>/*.txt, yields paths one at a time
      │                 (a generator — never a list)
      ▼
checkpoint.py           already done? → skip.  unchanged hash? → skip.
      │
      ▼
loaders/                bytes → Document        (encoding detection, validation)
      │
      ▼
parsers/                Document → [Section]    (ARTICLE / SECTION boundaries)
      │
      ▼
chunkers/               Section → [Chunk]       (sentence-aligned, ~400 tokens)
      │                                          + parent chunks at ~1,600
      ▼
[Chunk]  ──►  Phase 3 stores them
      │
      └── on any failure ──► dead_letter.py, and the run continues
```

---

## 2. File 1 — `src/ingestion/loaders/base.py`

### The Problem

Every loader — text, PDF, DOCX — needs the same five things: compute a content hash, generate a
deterministic `doc_id`, normalise whitespace, reject empty output, and build the `Document`. Only the
"turn bytes into a string" step differs. Duplicating the other five across four loaders means fixing
every bug four times.

### Design Decision

**A template-method base class.** The base implements `load()` as a fixed sequence of steps and leaves
exactly one step abstract: `_extract_text()`. Subclasses fill in the one part that varies. This is the
Template Method pattern, and it is the right shape whenever several implementations share an algorithm
but differ in one step.

The alternative — free functions plus a shared helper module — works, but nothing then enforces that a
new loader performs validation. With the template, a subclass *cannot* skip it, because it does not
control the sequence.

```python
from abc import abstractmethod
from pathlib import Path

from src.core.exceptions import DocumentLoadError
from src.core.interfaces import BaseDocumentLoader
from src.core.logging import logger
from src.core.models import Document, DocumentType
from src.core.utils import hash_text, make_doc_id, normalise_whitespace

#: Below this, a "document" is a stub — a header with no body, or a stray footer.
#: Cheaper to quarantine than to carry through embedding and storage.
MIN_CONTENT_CHARS = 200


class FileLoader(BaseDocumentLoader):
    """Template implementation shared by all file-based loaders.

    Subclasses override `_extract_text` (and `doc_type` / `extensions`). They do not
    override `load`, which fixes the order of operations so no implementation can
    accidentally skip validation.
    """

    #: File suffixes this loader claims, lowercase, with the dot.
    extensions: tuple[str, ...] = ()
    doc_type: DocumentType = DocumentType.TXT

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.extensions

    @abstractmethod
    def _extract_text(self, path: Path) -> str:
        """Format-specific: turn a file into a raw string. The only varying step."""

    def load(self, file_path: str) -> Document:
        """Read one file into a validated `Document`.

        Raises:
            DocumentLoadError: missing, empty, undecodable, or too short to be useful.
        """
        path = Path(file_path)

        if not path.is_file():
            raise DocumentLoadError("Not a file", details={"path": str(path)})

        # Check size before reading. A zero-byte file is common in scraped corpora,
        # and a 200 MB "contract" is a corpus error we should not pull into memory.
        size = path.stat().st_size
        if size == 0:
            raise DocumentLoadError("File is empty", details={"path": str(path)})
        if size > 100 * 1024 * 1024:
            raise DocumentLoadError(
                "File exceeds 100 MB", details={"path": str(path), "bytes": size}
            )

        try:
            raw = self._extract_text(path)
        except DocumentLoadError:
            raise                                   # already has context; do not re-wrap
        except Exception as exc:
            raise DocumentLoadError(
                f"{type(self).__name__} failed to extract text",
                details={"path": str(path)},
            ) from exc

        content = normalise_whitespace(raw)
        if len(content) < MIN_CONTENT_CHARS:
            raise DocumentLoadError(
                "Content too short to be a usable document",
                details={"path": str(path), "chars": len(content)},
            )

        logger.debug("Loaded document", extra={"path": str(path), "chars": len(content)})

        return Document(
            doc_id=make_doc_id(path),
            source_path=path.as_posix(),
            file_name=path.name,
            content=content,
            content_hash=hash_text(content),
            doc_type=self.doc_type,
            year=self._infer_year(path),
        )

    @staticmethod
    def _infer_year(path: Path) -> str | None:
        """Read the year from the parent directory name.

        The corpus is partitioned `contracts/<year>/`, so the directory *is* the
        metadata. Returned as a string; `Document`'s validator coerces to int and
        yields None for non-numeric names like `misc/`.
        """
        return path.parent.name
```

### Why hash the normalised content, not the raw bytes

`hash_file` (raw bytes) and `hash_text(content)` (normalised text) answer different questions, and
picking the wrong one causes needless re-work.

Raw-byte hashing changes if a file is re-saved with different line endings or re-encoded from `cp1252`
to UTF-8 — even though not a single word changed. Under byte hashing, that file looks modified and gets
fully re-embedded.

Normalised-text hashing is stable under exactly those cosmetic changes, because
`normalise_whitespace` has already collapsed them. So `content_hash` is the right fingerprint for
"has the *meaning* changed", which is the question the checkpoint needs to answer.

We still keep `hash_file` in `utils.py` — the checkpoint uses it as a **cheap** pre-filter, since
hashing bytes needs no decoding, parsing, or normalisation. The pattern is: byte hash to decide whether
to bother opening the document, content hash to decide whether to re-embed it.

### Failure Modes

**Every document rejected as "too short".** `MIN_CONTENT_CHARS = 200` is tuned for contracts. If you
point this at a corpus of short documents, lower it. Check the dead-letter directory before assuming
the loader is broken.

**`DocumentLoadError` wrapping a `DocumentLoadError`.** Prevented by the bare `raise` in the first
`except` clause. Without it, a subclass raising a precise error would have it buried inside a generic
one.

---

## 3. File 2 — `src/ingestion/loaders/txt_loader.py`

### The Problem

`open(path, encoding="utf-8")` fails on real EDGAR filings, and it is worth understanding why so you
recognise the symptom later.

Those documents were produced across three decades by hundreds of legal departments using Word on
Windows. The bytes are `cp1252` (Windows Western European), `latin-1`, occasionally UTF-16 with a BOM,
and sometimes genuinely mixed within a single file because someone pasted from two sources. Read
`cp1252` as UTF-8 and you get `UnicodeDecodeError` on the first smart quote — byte `0x92`, which is `'`
in `cp1252` and invalid as a standalone UTF-8 sequence.

Legal boilerplate is *full* of smart quotes, em dashes, and section symbols (`§`, byte `0xA7`). So this
is not an edge case; it is most of the corpus.

The prototype's handling was:

```python
try:
    with open(file_path, mode="r", encoding="utf-8") as f:
        text = f.read()
except Exception:
    continue                     # silently drops the document
```

Every `cp1252` file vanishes with no record. You would never know how much of your corpus is missing.

### Design Decision

**A cascade of decoders, most-likely first, with a lossy last resort.** Try UTF-8 (correct for modern
files and fails fast when wrong), then check for a BOM, then `cp1252`, then `latin-1`. `latin-1` is the
important backstop: it maps all 256 byte values to *some* character, so **it can never raise**. Worst
case you get a few odd glyphs instead of losing the document.

**Why not `chardet` / chardetect?** Statistical encoding detection is slow — it samples and scores
candidate encodings — and across 650k files that cost is enormous. Our cascade is deterministic, and
`cp1252` versus `latin-1` differ only in the `0x80–0x9F` range, so a wrong guess between them is
cosmetic rather than fatal. We do use a cheap heuristic to *detect* the failure and log it.

**Read bytes once, decode in memory.** Re-opening the file per attempt means up to four disk reads.
Read once, then try decoding the same buffer repeatedly.

```python
from pathlib import Path

from src.core.exceptions import DocumentLoadError
from src.core.logging import logger
from src.core.models import DocumentType

from .base import FileLoader

#: Ordered decode attempts. utf-8 first because it is correct for modern files and
#: fails fast when wrong. latin-1 last because it accepts any byte and cannot raise,
#: which guarantees we never lose a document to encoding alone.
_ENCODINGS: tuple[str, ...] = ("utf-8", "cp1252", "latin-1")

#: Byte-order marks, checked before the cascade because they are unambiguous.
_BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)

#: Sequences that appear when cp1252 bytes are decoded as latin-1. Their presence
#: means we fell through to the lossy backstop and the text has mojibake.
_MOJIBAKE_MARKERS = ("â€™", "â€œ", "â€\x9d", "Â§", "â€”")


class TextDocumentLoader(FileLoader):
    """Loads plain-text documents, tolerating the encoding chaos of real corpora."""

    extensions = (".txt", ".text", ".md")
    doc_type = DocumentType.TXT

    def _extract_text(self, path: Path) -> str:
        raw = path.read_bytes()

        # 1. An explicit BOM is definitive — trust it over any heuristic.
        for bom, encoding in _BOMS:
            if raw.startswith(bom):
                return raw.decode(encoding, errors="replace")

        # 2. Reject binaries masquerading as .txt before wasting decode attempts.
        #    A NUL byte in the first 8 KB means this is not text.
        if b"\x00" in raw[:8192]:
            raise DocumentLoadError(
                "Binary content in a text file", details={"path": str(path)}
            )

        # 3. Cascade. The final encoding cannot fail, so this always returns.
        for encoding in _ENCODINGS:
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError:
                continue

            if encoding != "utf-8":
                logger.debug(
                    "Non-UTF-8 document decoded",
                    extra={"path": str(path), "encoding": encoding},
                )
            if encoding == "latin-1" and self._has_mojibake(text):
                # Not fatal — the text is readable and the clauses are intact — but
                # worth counting, because a high rate means the cascade needs tuning.
                logger.warning(
                    "Probable mojibake after latin-1 fallback",
                    extra={"path": str(path)},
                )
            return text

        # Unreachable: latin-1 accepts every byte sequence. Kept so that a future
        # edit to _ENCODINGS that removes the safe backstop fails loudly.
        raise DocumentLoadError(
            "All decoders failed", details={"path": str(path), "tried": list(_ENCODINGS)}
        )

    @staticmethod
    def _has_mojibake(text: str, sample_chars: int = 4000) -> bool:
        sample = text[:sample_chars]
        return any(marker in sample for marker in _MOJIBAKE_MARKERS)
```

### The Theory: what mojibake actually is

Worth understanding once, because you will see it and need to diagnose it in seconds rather than hours.

The right single quote `'` is Unicode U+2019. In `cp1252` it is the single byte `0x92`. In UTF-8 it is
three bytes: `0xE2 0x80 0x99`.

Now suppose a file is genuinely UTF-8 but you decode it as `latin-1`. `latin-1` maps every byte to the
character with that code point, one byte at a time. So those three bytes become three separate
characters: `â` (0xE2), `€` (0x80), `™` (0x99) — the string `â€™`.

That is why `â€™` is the classic signature of "UTF-8 read as latin-1", and why we scan for it.
Its presence means the cascade reached the lossy backstop on a file that was actually valid UTF-8 —
which should not happen, since UTF-8 is tried first, unless the file is *mixed* encoding. A high
mojibake rate is therefore a signal that a chunk of the corpus has genuinely mixed bytes and may need
per-line decoding.

The text stays perfectly usable for retrieval either way. `â€™` inside a clause does not stop BM25
from matching "termination" or a bi-encoder from placing the passage correctly. It just looks wrong in
a citation, which is why we log rather than reject.

### Failure Modes

**Mojibake warnings on thousands of files.** Investigate before ignoring. If the affected files share
a year directory, that batch was scraped differently and may warrant its own encoding.

**`Binary content in a text file`** — usually a `.txt` that is really a PDF or a zip. Correctly
quarantined; check a few in the dead-letter directory to confirm.

**A document loads but the text is one enormous line.** Not an encoding problem —
`normalise_whitespace` in the base class collapses newlines by design, and the parser works on the
pre-normalised structure. If your parser finds no sections, this is why: see §5's note on
newline-sensitive parsing.

---

## 4. File 3 — `src/ingestion/loaders/factory.py`

### The Problem

The pipeline holds a path and needs the right loader. Writing
`if path.endswith(".pdf"): ... elif path.endswith(".docx"): ...` inside `pipeline.py` means the
orchestrator must know every format, and Phase 14 (which adds PDF and OCR loaders) has to edit the
orchestrator to add one.

### Design Decision

**A registry keyed by the loaders' own `supports()` method.** The factory holds a list of instances and
asks each whether it can handle the path. Adding a format in Phase 14 means appending to one list;
nothing else changes. This is the Open/Closed Principle in its most literal form — open to extension,
closed to modification.

Note the factory returns *shared instances*, not new objects per call. Loaders are stateless, and
Phase 14's PDF loader will hold an expensive parser handle that should not be rebuilt 650,000 times.

```python
from pathlib import Path

from src.core.exceptions import DocumentLoadError
from src.core.interfaces import BaseDocumentLoader
from src.core.models import Document

from .txt_loader import TextDocumentLoader


class LoaderFactory:
    """Dispatches a file path to the loader that claims it.

    Registration order is precedence order: the first loader whose `supports()`
    returns True wins.
    """

    def __init__(self) -> None:
        self._loaders: list[BaseDocumentLoader] = [
            TextDocumentLoader(),
            # Phase 14 appends PdfDocumentLoader(), DocxDocumentLoader(), HtmlDocumentLoader()
        ]

    def register(self, loader: BaseDocumentLoader, *, first: bool = False) -> None:
        """Add a loader. `first=True` gives it precedence over existing ones."""
        if first:
            self._loaders.insert(0, loader)
        else:
            self._loaders.append(loader)

    def get_loader(self, file_path: str) -> BaseDocumentLoader:
        for loader in self._loaders:
            if loader.supports(file_path):
                return loader
        raise DocumentLoadError(
            "No loader registered for this file type",
            details={"path": file_path, "suffix": Path(file_path).suffix},
        )

    def load(self, file_path: str) -> Document:
        """Convenience: resolve the loader and load in one call."""
        return self.get_loader(file_path).load(file_path)

    @property
    def supported_extensions(self) -> set[str]:
        """Every extension any registered loader claims. Used by discovery to filter."""
        exts: set[str] = set()
        for loader in self._loaders:
            exts.update(getattr(loader, "extensions", ()))
        return exts
```

### Failure Modes

**`No loader registered`** for files you expected to work — check the extension is in
`TextDocumentLoader.extensions`. EDGAR sometimes uses `.TXT` uppercase, which is why `supports()`
lowercases the suffix.

**The factory constructed inside a loop.** Build it once and pass it. Constructing it per document is
harmless today and expensive after Phase 14.

---

## 5. File 4 — `src/ingestion/parsers/legal_parser.py`

### The Problem

This is the heart of the phase, and the reason a legal RAG system can outperform a generic one.

A naive chunker slices every 400 tokens with no regard for meaning. Applied to a contract, it produces
chunks like:

> *"…in no event shall Seller's aggregate liability exceed"*

The chunk ends mid-clause. The number — the single fact the user asked for — is in the next chunk.
Retrieval returns this passage because it matches "liability", the LLM sees no figure, and either says
it does not know or, worse, borrows a number from an adjacent unrelated clause.

Contracts have explicit machine-readable structure: `ARTICLE IV`, `SECTION 7.02`, `EXHIBIT B`. Parsing
along those boundaries and chunking *within* them means a chunk never spans two clauses. It also gives
every chunk a `section_title`, which is what makes a citation like
`[SOURCE 1: Credit Agreement, SECTION 2.01]` possible.

### Design Decision

**Regex, not an LLM and not a trained model.** This surprises people, so the reasoning matters. LLM-based
structural extraction would cost 650,000 API calls, take days, and be non-deterministic. A trained
layout model needs labelled data we do not have. Contract section headers are *lexically marked* —
they literally start with the word `SECTION` followed by a number — which is precisely the class of
problem regular expressions are good at. Use the cheapest tool that solves it.

**Parse the raw content, not the normalised content.** Section headers are usually on their own line,
and `normalise_whitespace` collapses line structure. So the parser must run on text that still has
newlines. This means `Document.content` cannot be the fully-flattened version — we normalise
*horizontal* whitespace only (spaces and tabs) and preserve newlines. Check `normalise_whitespace` in
Phase 1: its final regex is `[ \t\r\f\v]+`, which deliberately excludes `\n`. That was not an accident.

**Header candidates must be short.** The string `SECTION 7.02` appears both as a header and inside
cross-references: *"…as further described in SECTION 7.02 hereof…"*. A header is a short standalone
line; a cross-reference sits inside a long sentence. A length ceiling separates them with one cheap
check, and gets the overwhelming majority right.

**Fall back to a single section rather than failing.** Many documents have no recognisable headers —
short exhibits, letter agreements, amendments. Those are still valuable content. A parser that raises
on them would quarantine a large slice of the corpus for no good reason, so unstructured documents
become one `Preamble` section and proceed normally.

```python
import re

from src.core.logging import logger
from src.core.interfaces import BaseParser
from src.core.models import Document, Section
from src.core.utils import make_section_id

#: Header patterns, tried in order. Each must match at the START of a line.
#:
#: Ordering matters: the numbered SECTION/ARTICLE forms are checked before the
#: bare-keyword form, so "SECTION 5. TERMINATION" is captured as a numbered section
#: rather than matching the standalone TERMINATION rule.
_HEADER_PATTERNS: tuple[re.Pattern[str], ...] = (
    # SECTION 7.02  |  Section 7.02.  |  SECTION 12
    re.compile(r"^(SECTION\s+\d+(?:\.\d+)*\.?)\s*(.*)$", re.IGNORECASE),
    # ARTICLE IV  |  ARTICLE 4  |  Article Fourth
    re.compile(r"^(ARTICLE\s+(?:[IVXLCDM]+|\d+|[A-Z][a-z]+)\.?)\s*(.*)$", re.IGNORECASE),
    # EXHIBIT B  |  SCHEDULE 2  |  ANNEX I  |  APPENDIX A
    re.compile(
        r"^((?:EXHIBIT|SCHEDULE|ANNEX|APPENDIX)\s+(?:[A-Z0-9]+|[IVXLCDM]+)\.?)\s*(.*)$",
        re.IGNORECASE,
    ),
    # 7.02  Limitation of Liability      (numeric-only header, common in older filings)
    re.compile(r"^(\d+\.\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z\s,'\-]{3,60})$"),
    # Standalone all-caps clause keywords on their own line
    re.compile(
        r"^(INDEMNIFICATION|INDEMNITY|TERMINATION|CONFIDENTIALITY|GOVERNING\s+LAW"
        r"|LIMITATION\s+OF\s+LIABILITY|REPRESENTATIONS\s+AND\s+WARRANTIES"
        r"|FORCE\s+MAJEURE|ASSIGNMENT|NOTICES|SEVERABILITY|ENTIRE\s+AGREEMENT"
        r"|DEFINITIONS|PAYMENT\s+TERMS|WARRANTY|ARBITRATION|DISPUTE\s+RESOLUTION)"
        r"\s*[:.]?\s*$",
        re.IGNORECASE,
    ),
)

#: A header is a short standalone line. Longer lines containing "SECTION 7.02" are
#: cross-references inside prose, not headers.
MAX_HEADER_CHARS = 90

#: Discard sections shorter than this — almost always a header immediately followed
#: by another header, i.e. a table of contents entry rather than real content.
MIN_SECTION_CHARS = 120


class LegalSectionParser(BaseParser):
    """Segments a contract along ARTICLE / SECTION / EXHIBIT boundaries."""

    def parse(self, document: Document) -> list[Section]:
        lines = document.content.split("\n")

        current_title = "Preamble"
        current_body: list[str] = []
        collected: list[tuple[str, str]] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            header = self._match_header(stripped)
            if header is None:
                current_body.append(stripped)
                continue

            # A header closes the previous section and opens a new one.
            collected.append((current_title, "\n".join(current_body)))
            current_title = header

            # Some headers carry inline body text: "SECTION 3.1 Term. This Agreement
            # shall commence..." — the trailing prose belongs to the new section.
            current_body = []
            inline = self._inline_remainder(stripped, header)
            if inline:
                current_body.append(inline)

        collected.append((current_title, "\n".join(current_body)))

        sections = self._build_sections(document, collected)

        if not sections:
            # No recognisable structure. Rather than quarantine perfectly good text,
            # treat the whole document as one section. Chunking still works; only
            # citation precision is reduced.
            logger.debug(
                "No sections detected; treating document as single block",
                extra={"doc_id": document.doc_id, "file": document.file_name},
            )
            return [
                Section(
                    section_id=make_section_id(document.doc_id, "Preamble", 0),
                    doc_id=document.doc_id,
                    title="Preamble",
                    text=document.content,
                    order=0,
                )
            ]

        logger.debug(
            "Parsed sections",
            extra={"doc_id": document.doc_id, "sections": len(sections)},
        )
        return sections

    # ─── internals ─────────────────────────────────────────────────────────

    def _match_header(self, line: str) -> str | None:
        """Return a normalised header title, or None if this line is body text."""
        if len(line) > MAX_HEADER_CHARS:
            return None

        for pattern in _HEADER_PATTERNS:
            match = pattern.match(line)
            if not match:
                continue

            label = match.group(1).strip().rstrip(".")
            caption = (match.group(2) if match.lastindex and match.lastindex >= 2 else "").strip()

            # Keep a short caption ("SECTION 7.02 Limitation of Liability") because it
            # makes citations readable. Drop a long one — that is body prose bleeding in.
            if caption and len(caption) <= 60:
                # Strip trailing sentence text: keep "Term" from "Term. This Agreement..."
                caption = re.split(r"(?<=[a-z])\.\s", caption)[0].strip(" .:")
                if caption:
                    return f"{label} {caption}"[:200]
            return label[:200]

        return None

    def _inline_remainder(self, line: str, header: str) -> str:
        """Body text that appeared on the same line as the header."""
        # Compare case-insensitively; the header we return may be re-cased.
        lowered = line.lower()
        label = header.split()[0].lower()
        idx = lowered.find(label)
        if idx == -1:
            return ""
        remainder = line[idx + len(header) :].strip(" .:\t")
        return remainder if len(remainder) > 40 else ""

    def _build_sections(
        self, document: Document, collected: list[tuple[str, str]]
    ) -> list[Section]:
        sections: list[Section] = []
        order = 0
        for title, body in collected:
            if len(body.strip()) < MIN_SECTION_CHARS:
                continue
            sections.append(
                Section(
                    section_id=make_section_id(document.doc_id, title, order),
                    doc_id=document.doc_id,
                    title=title,
                    text=body,
                    order=order,
                )
            )
            order += 1
        return sections
```

### The Theory: why `section_id` includes `order`

`make_section_id(doc_id, title, order)` takes the ordinal as well as the title, and the reason is a
correctness bug you would otherwise hit.

Contracts repeat titles. A long agreement can have `EXHIBIT A` and, in an amendment appended to the
same file, a second `EXHIBIT A`. It may have `TERMINATION` under the services article and again under
the licence article. Keying purely on `(doc_id, title)` would give both the identical `section_id`.

Two different sections sharing an ID means their chunks collide too, because `make_chunk_id` derives
from `section_id`. The second section's chunks would silently overwrite the first's — data loss with no
error. Including `order` breaks the tie, since two sections cannot occupy the same position.

The cost: `section_id` is now sensitive to position, so inserting a new section shifts every subsequent
ID. That is exactly the case §"the limit of this" in Phase 1 warned about, and exactly why
`delete_by_doc_id` before upsert is mandatory rather than optional.

### Failure Modes

**Every document yields one `Preamble` section.** The most common Phase 2 problem, and almost always
one of two causes. Either `normalise_whitespace` is stripping newlines — verify its final regex is
`[ \t\r\f\v]+` and not `\s+`, since `\s` includes `\n` and would flatten the document into one line
with no line starts for `^` to anchor to. Or the corpus genuinely uses a header convention not in
`_HEADER_PATTERNS`; print a few documents' first 50 lines and look.

**Sections split at cross-references.** `MAX_HEADER_CHARS` is too high, or a reference sits alone on a
short line. Lower the ceiling toward 70 and re-check.

**Thousands of one-line sections.** You are parsing a table of contents, where every line looks like a
header. `MIN_SECTION_CHARS` filters these; raise it if they persist.

---

## 6. File 5 — `src/ingestion/parsers/metadata_extractor.py`

### The Problem

A citation reading *"according to document 4f8a2c1e"* is useless. It must read *"according to the 2003
Applied Materials Supply Agreement, SECTION 7.02"*. That requires a real contract name.

The prototype took the first non-empty line:

```python
contract_name = lines[0] if lines else "Unknown Contract"
```

On EDGAR filings the first line is typically `EX-10.1`, `Exhibit 10.1`, or a page number — an SEC
exhibit label, not a title. So nearly every citation would say `EX-10.1`.

### Design Decision

**Scan a window of early lines and score candidates**, rather than trusting position. Real titles have
recognisable properties: they contain the word "AGREEMENT" or "CONTRACT", they are title-cased or
all-caps, they are 20–200 characters, and they are not boilerplate like "Table of Contents".

**Skip a known blacklist first.** Exhibit labels, page markers, and confidentiality stamps are
predictable and can be excluded cheaply.

**Never fail.** Metadata is best-effort. A missing title degrades citation quality; it must not
quarantine the document.

```python
import re

from src.core.models import Document

#: How many leading lines to consider. Titles appear early or not at all.
_SCAN_LINES = 60

#: Lines that are never titles.
_BLACKLIST = re.compile(
    r"^(?:ex(?:hibit)?[\s\-]*\d|page\s+\d|table\s+of\s+contents|confidential"
    r"|execution\s+(?:copy|version)|draft|\[?\s*redacted|\d+\s*$|-+\s*$)",
    re.IGNORECASE,
)

#: Words that make a line likely to be a contract title.
_TITLE_KEYWORDS = (
    "agreement", "contract", "indenture", "lease", "license", "licence",
    "amendment", "guaranty", "guarantee", "note", "deed", "assignment",
    "memorandum", "certificate", "plan", "warrant", "waiver",
)

_PARTY_PATTERN = re.compile(
    r"\b(?:between|among|by\s+and\s+between)\b(.{10,400}?)"
    r"(?:\.|;|\bWHEREAS\b|\bRECITALS\b)",
    re.IGNORECASE | re.DOTALL,
)

_ENTITY_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9&.,'\- ]{2,60}?"
    r"(?:Inc|Corp|Corporation|Company|Co|LLC|L\.L\.C|Ltd|Limited|LP|L\.P|PLC|N\.V|GmbH|S\.A)\.?)"
)

_DATE_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")


class ContractMetadataExtractor:
    """Best-effort extraction of title, parties, and year from contract text.

    Not a `BaseParser` — it does not produce Sections. It enriches a `Document`.
    """

    def enrich(self, document: Document) -> Document:
        """Return a copy of the document with metadata filled in.

        `Document` is frozen, so this returns a new instance via `model_copy`
        rather than mutating in place.
        """
        head = document.content[:20_000]
        lines = [ln.strip() for ln in head.split("\n") if ln.strip()][:_SCAN_LINES]

        title = self._extract_title(lines) or document.contract_name
        parties = self._extract_parties(head)
        year = document.year or self._extract_year(head)

        metadata = dict(document.metadata)
        if parties:
            metadata["parties"] = parties

        return document.model_copy(
            update={"contract_name": title, "year": year, "metadata": metadata}
        )

    # ─── internals ─────────────────────────────────────────────────────────

    def _extract_title(self, lines: list[str]) -> str | None:
        best: tuple[int, str] | None = None

        for position, line in enumerate(lines):
            if _BLACKLIST.match(line) or not 12 <= len(line) <= 200:
                continue

            score = self._score_title(line, position)
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, line)

        if best is None:
            return None
        return re.sub(r"\s+", " ", best[1]).strip(" .:-")[:300]

    @staticmethod
    def _score_title(line: str, position: int) -> int:
        score = 0
        lowered = line.lower()

        if any(keyword in lowered for keyword in _TITLE_KEYWORDS):
            score += 10
        else:
            # Without a document-type keyword it is probably not a title at all.
            return 0

        letters = [c for c in line if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
            score += 4                        # ALL CAPS — very common for titles
        elif line.istitle():
            score += 2

        # Earlier lines are likelier titles, but position is a tiebreak, not the rule.
        score += max(0, 5 - position // 4)

        if line.endswith((".", ";", ",")):
            score -= 3                        # sentences end in punctuation; titles rarely do
        if len(line.split()) > 25:
            score -= 4                        # too long to be a heading

        return score

    @staticmethod
    def _extract_parties(text: str) -> list[str]:
        match = _PARTY_PATTERN.search(text)
        region = match.group(1) if match else text[:3000]

        seen: list[str] = []
        for entity in _ENTITY_PATTERN.findall(region):
            cleaned = re.sub(r"\s+", " ", entity).strip(" ,.")
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
            if len(seen) >= 6:
                break
        return seen

    @staticmethod
    def _extract_year(text: str) -> int | None:
        """Earliest plausible year in the opening text — usually the execution date."""
        years = [int(y) for y in _DATE_YEAR.findall(text[:4000])]
        return min(years) if years else None
```

### Failure Modes

**Titles are still `EX-10.1`.** Add the pattern to `_BLACKLIST`. Print the top ten scoring lines for a
few documents to see what is winning.

**Empty `parties` on most documents.** The `between … among` construction is common but not universal.
This is best-effort metadata; do not spend hours on it. Phase 13 extracts entities properly with an
LLM, and that is the right tool for the job.

**Extracted year contradicts the directory year.** We prefer the directory (`document.year or ...`)
because a filing directory is authoritative while an in-text year might be a referenced prior
agreement.

---

## 7. File 6 — `src/ingestion/chunkers/sentence_chunker.py`

### The Problem

Two problems, and the second is subtle.

**Splitting on `". "` is wrong for legal text.** The prototype does `text.split(". ")`, which shatters
on the abbreviations legal writing is built from: `Inc.`, `Ltd.`, `L.P.`, `U.S.`, `No.`, `Art.`,
`e.g.`, `i.e.`, `et seq.`, and section numbers like `7.02`. *"Acme Inc. shall indemnify"* becomes two
"sentences", and chunk boundaries land mid-clause.

**Chunk boundaries lose context.** Even with perfect sentence detection, a clause split across two
chunks means neither chunk is independently answerable. The fix is overlap: repeat the last couple of
sentences of chunk N at the start of chunk N+1, so a fact near a boundary appears whole in at least one
chunk.

The prototype attempts overlap with `i = max(j - config.SENTENCE_OVERLAP, i + 1)`, which mostly works
but tangles with its oversized-sentence branch — in some paths it advances `i` past sentences that were
never emitted, silently dropping text.

### Design Decision

**An abbreviation-aware regex splitter, not `nltk` or `spaCy`.** Both are excellent and both are
overkill: `nltk.punkt` needs a downloaded model and `spaCy` loads a pipeline costing hundreds of MB and
significant per-document time. Across 650k documents that is hours. A regex plus an abbreviation set
captures the cases that matter here, deterministically and fast.

**Split on the boundary, then repair.** Rather than trying to write one regex that handles every case,
we split aggressively on sentence-ending punctuation and then *merge back* fragments whose break was
caused by a known abbreviation or an initial. Splitting then repairing is far easier to reason about
and extend than a single monstrous pattern.

**Guarantee forward progress explicitly.** The chunk loop asserts that its cursor always advances. An
off-by-one in an overlap calculation otherwise produces an infinite loop that hangs the ingestion with
no error — the worst failure mode available, because at 650k documents you will assume it is just slow.

```python
import re

from src.core.exceptions import ChunkingError
from src.core.interfaces import BaseChunker
from src.core.models import Chunk, ChunkLevel, Document, Section
from src.core.utils import count_tokens, make_chunk_id

#: Tokens ending in a period that do NOT end a sentence.
_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        # corporate forms
        "inc", "corp", "co", "ltd", "llc", "llp", "lp", "plc", "gmbh", "sa", "nv", "ag",
        # honorifics
        "mr", "mrs", "ms", "dr", "prof", "hon", "esq", "jr", "sr",
        # legal and citation
        "no", "nos", "art", "arts", "sec", "secs", "cl", "para", "paras", "pp", "vol",
        "ch", "sched", "ex", "fig", "seq", "supra", "infra", "cf", "id", "ibid", "viz",
        "et", "al", "eg", "ie", "etc", "approx", "aka",
        # geographic and temporal
        "us", "usa", "uk", "eu", "st", "ave", "rd", "blvd", "dept",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    }
)

#: Candidate boundary: sentence punctuation, whitespace, then something that looks
#: like a new sentence — a capital, a quote, or an opening bracket.
_BOUNDARY = re.compile(r'(?<=[.!?])["\')\]]*\s+(?=["\'(\[]*[A-Z0-9])')

#: Trailing token of a fragment, without its final period.
_LAST_TOKEN = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")

#: A single capital letter plus period — an initial, as in "J. Smith".
_INITIAL = re.compile(r"\b[A-Z]\.$")

#: A number ending in a period — "7.02." or "Section 5." — the sentence continues.
_TRAILING_NUMBER = re.compile(r"\b\d+(?:\.\d+)*\.$")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, tolerant of legal abbreviations.

    Strategy: split aggressively on punctuation boundaries, then merge back any
    fragment whose break was caused by an abbreviation, an initial, or a number.
    Splitting and repairing is far easier to extend than one exhaustive regex.
    """
    text = text.strip()
    if not text:
        return []

    fragments = _BOUNDARY.split(text)
    if len(fragments) == 1:
        return [text]

    sentences: list[str] = []
    buffer = fragments[0]

    for fragment in fragments[1:]:
        if _is_false_boundary(buffer):
            buffer = f"{buffer} {fragment}"       # not a real break — rejoin
        else:
            sentences.append(buffer.strip())
            buffer = fragment

    if buffer.strip():
        sentences.append(buffer.strip())

    return [s for s in sentences if s]


def _is_false_boundary(fragment: str) -> bool:
    """Whether `fragment` ends in something that only looks like a sentence end."""
    tail = fragment.rstrip()

    if _INITIAL.search(tail) or _TRAILING_NUMBER.search(tail):
        return True

    match = _LAST_TOKEN.search(tail)
    if match:
        token = match.group(1).replace(".", "").lower()
        if token in _ABBREVIATIONS:
            return True
        # A single letter, e.g. the "(a)" enumerations common in contracts.
        if len(token) == 1:
            return True

    return False


class SentenceChunker(BaseChunker):
    """Groups sentences into token-bounded chunks with sentence-level overlap."""

    def __init__(
        self,
        max_tokens: int | None = None,
        overlap_sentences: int | None = None,
        min_tokens: int = 30,
    ) -> None:
        from config.settings import settings

        self.max_tokens = max_tokens or settings.MAX_TOKENS_PER_CHUNK
        self.overlap_sentences = (
            overlap_sentences if overlap_sentences is not None else settings.SENTENCE_OVERLAP
        )
        self.min_tokens = min_tokens

        if self.overlap_sentences < 0:
            raise ChunkingError("overlap_sentences must be >= 0")

    def chunk(self, section: Section, document: Document) -> list[Chunk]:
        texts = self.split_text(section.text)

        chunks: list[Chunk] = []
        for index, text in enumerate(texts):
            tokens = count_tokens(text)
            if tokens < self.min_tokens:
                continue                          # fragment too small to be useful alone
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.doc_id, section.section_id, index),
                    doc_id=document.doc_id,
                    section_id=section.section_id,
                    text=text,
                    chunk_index=index,
                    token_count=tokens,
                    contract_name=document.contract_name,
                    file_name=document.file_name,
                    section_title=section.title,
                    year=document.year,
                    chunk_level=ChunkLevel.STANDALONE,
                )
            )
        return chunks

    def split_text(self, text: str) -> list[str]:
        """Group sentences into ~max_tokens chunks, overlapping by N sentences."""
        sentences = split_sentences(text)
        if not sentences:
            return []

        # Pre-compute token counts once. Tokenizing inside the loop would re-encode
        # every overlapped sentence on each pass.
        lengths = [count_tokens(s) for s in sentences]

        chunks: list[str] = []
        start = 0
        total = len(sentences)

        while start < total:
            # A single sentence larger than the budget cannot be grouped; split it
            # on tokens directly and move past it.
            if lengths[start] > self.max_tokens:
                chunks.extend(self._split_oversized(sentences[start]))
                start += 1
                continue

            end = start
            budget = 0
            while end < total and budget + lengths[end] <= self.max_tokens:
                budget += lengths[end]
                end += 1

            chunks.append(" ".join(sentences[start:end]))

            if end >= total:
                break

            # Step back by the overlap so the next chunk repeats trailing context.
            next_start = end - self.overlap_sentences

            # Forward progress is mandatory. With a large overlap and a small chunk,
            # next_start can land at or before start, which loops forever. At 650k
            # documents an infinite loop reads as "still running" and costs you hours.
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return chunks

    def _split_oversized(self, sentence: str) -> list[str]:
        """Hard-split a single sentence that exceeds the token budget.

        Rare — usually a run-on definitions clause or a table flattened into prose.
        We split on tokens rather than characters so each piece respects the budget.
        """
        from src.core.utils import get_tokenizer

        tokenizer = get_tokenizer()
        token_ids = tokenizer.encode(sentence, disallowed_special=())
        return [
            tokenizer.decode(token_ids[i : i + self.max_tokens])
            for i in range(0, len(token_ids), self.max_tokens)
        ]
```

### The Theory: why overlap works, and what it costs

Suppose the fact you need sits at the boundary between chunk 3 and chunk 4:

```text
chunk 3: "... Seller's liability shall be limited. The aggregate cap"
chunk 4: "shall not exceed $2,000,000 in any twelve-month period. ..."
```

Neither chunk answers *"what is the liability cap?"* Chunk 3 has the subject without the number; chunk
4 has the number without the subject. Retrieval will probably surface chunk 3, since it matches
"liability", and the LLM has no figure to cite.

With a 2-sentence overlap, chunk 4 begins by repeating the tail of chunk 3, so the complete statement
exists intact in at least one chunk.

The cost is real and worth stating: overlap duplicates text. A 2-sentence overlap on 5-sentence chunks
inflates your index by roughly 40% — more storage, more embedding compute, and a chance the same
passage appears twice in your top-k, wasting a context slot. That last problem is what Reciprocal Rank
Fusion's de-duplication in Phase 4 exists to clean up.

`SENTENCE_OVERLAP = 2` is a reasonable default. If retrieval returns near-duplicate chunks constantly,
lower it. If answers keep missing figures that sit near boundaries, raise it.

### Failure Modes

**Ingestion hangs with no error and no progress.** Almost certainly an overlap loop, which is why the
`next_start <= start` guard exists. If you altered the loop, restore that check.

**Chunks split at `Inc.` or `Section 7.02`.** An abbreviation is missing from `_ABBREVIATIONS`, or
`_TRAILING_NUMBER` was removed. Test `split_sentences` on a real paragraph directly.

**Chunks far under `max_tokens`.** Expected. A chunk closes when the *next* sentence would overflow, so
average size lands well below the ceiling. It is a ceiling, not a target.

**Many chunks dropped by `min_tokens`.** Signature/notary blocks and TOC fragments. Check the counts
are plausible before raising the threshold.

---

## 8. File 7 — `src/ingestion/chunkers/parent_child_chunker.py`

### The Problem

Chunk size is a genuine dilemma with no single right answer.

**Small chunks retrieve better.** A 400-token passage about exactly one clause produces a focused
embedding. A 2,000-token chunk covering five clauses produces a vector that is the average of five
topics and is strongly matched by nothing.

**Large chunks generate better.** Give the LLM 400 tokens and it often lacks the surrounding
definitions, the cross-referenced subsection, or the proviso that qualifies the sentence it is reading.

You cannot optimise both with one size.

### Design Decision

**Two levels: search the small, generate from the large.** Build ~1,600-token **parent** chunks, split
each into ~400-token **children**, embed and search the children, then substitute each matched child's
parent before generation. You get the retrieval precision of small chunks and the generative context of
large ones.

**Where do parents live?** Three options were considered. Storing the parent text inside every child's
payload is simplest but duplicates each parent roughly four times, which is expensive at this corpus
size. A separate document store (SQLite, or Redis) adds a component, and Redis is the wrong home
because Phase 1 configured it with `allkeys-lru` eviction — parents would silently disappear.

So: **parents live in the same collection as children, distinguished by `chunk_level`.** Search filters
to `chunk_level == "child"`; parents are fetched by ID. This needs no new infrastructure, keeps one
source of truth, and sets up Phase 16 directly, since RAPTOR adds `SUMMARY` nodes as a third level in
the same scheme. This is why `ChunkLevel` exists on the `Chunk` model.

```python
from src.core.interfaces import BaseChunker
from src.core.logging import logger
from src.core.models import Chunk, ChunkLevel, Document, Section
from src.core.utils import count_tokens, make_chunk_id

from .sentence_chunker import SentenceChunker


class ParentChildChunker(BaseChunker):
    """Two-level chunking: large parents for context, small children for retrieval.

    Emits BOTH levels in one list. Phase 3 indexes them all; Phase 4 filters
    searches to `chunk_level == CHILD` and swaps in parents before generation.
    """

    def __init__(
        self,
        parent_tokens: int = 1600,
        child_tokens: int | None = None,
        child_overlap: int | None = None,
    ) -> None:
        from config.settings import settings

        self.parent_tokens = parent_tokens
        self.child_tokens = child_tokens or settings.MAX_TOKENS_PER_CHUNK

        if self.child_tokens >= self.parent_tokens:
            raise ValueError(
                f"child_tokens ({self.child_tokens}) must be smaller than "
                f"parent_tokens ({self.parent_tokens}); otherwise every parent has "
                "exactly one child and the hierarchy is pointless"
            )

        # Parents do not overlap: they are context containers, not retrieval targets,
        # and overlapping them would duplicate a lot of text for no benefit.
        self._parent_splitter = SentenceChunker(
            max_tokens=self.parent_tokens, overlap_sentences=0
        )
        self._child_splitter = SentenceChunker(
            max_tokens=self.child_tokens, overlap_sentences=child_overlap
        )

    def chunk(self, section: Section, document: Document) -> list[Chunk]:
        parent_texts = self._parent_splitter.split_text(section.text)
        if not parent_texts:
            return []

        chunks: list[Chunk] = []

        # Child indexes are unique across the whole section, not per parent, so that
        # make_chunk_id never produces a collision between two parents' children.
        child_counter = 0

        for parent_index, parent_text in enumerate(parent_texts):
            parent_id = make_chunk_id(
                document.doc_id, section.section_id, self._parent_slot(parent_index)
            )

            chunks.append(
                self._build(
                    document, section, parent_id, parent_text,
                    index=self._parent_slot(parent_index),
                    level=ChunkLevel.PARENT, parent_of=None,
                )
            )

            for child_text in self._child_splitter.split_text(parent_text):
                chunks.append(
                    self._build(
                        document, section, 
                        make_chunk_id(document.doc_id, section.section_id, child_counter),
                        child_text,
                        index=child_counter,
                        level=ChunkLevel.CHILD, parent_of=parent_id,
                    )
                )
                child_counter += 1

        logger.debug(
            "Parent-child chunked",
            extra={
                "doc_id": document.doc_id,
                "section": section.title,
                "parents": len(parent_texts),
                "children": child_counter,
            },
        )
        return chunks

    @staticmethod
    def _parent_slot(parent_index: int) -> int:
        """Index space reserved for parents.

        Parents and children share one `chunk_index` namespace because
        `make_chunk_id` keys on it. Offsetting parents by a large constant keeps the
        two families from ever colliding, however many children a section produces.
        """
        return 1_000_000 + parent_index

    def _build(
        self, document: Document, section: Section, chunk_id: str, text: str,
        *, index: int, level: ChunkLevel, parent_of: str | None,
    ) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            doc_id=document.doc_id,
            section_id=section.section_id,
            text=text,
            chunk_index=index,
            token_count=count_tokens(text),
            contract_name=document.contract_name,
            file_name=document.file_name,
            section_title=section.title,
            year=document.year,
            chunk_level=level,
            parent_id=parent_of,
        )
```

### Why the parent index offset matters

`make_chunk_id(doc_id, section_id, chunk_index)` is a pure function of three values, so two chunks in
the same section with the same `chunk_index` get the same ID — and one silently overwrites the other.

Parents and children both need indexes, and both live in the same section. Numbering parents `0, 1, 2`
and children `0, 1, 2, …` would collide immediately. Offsetting parents to `1_000_000+` guarantees
separation for any realistic section size.

A cleaner alternative would be to include the level in the ID derivation —
`make_chunk_id(doc_id, section_id, index, level)`. That is arguably better design, and the reason we did
not is that it changes a Phase 1 signature that Phases 3 and 15 already reference. The offset achieves
the same guarantee without touching the frozen contract. Worth knowing as a real engineering tradeoff:
a slightly uglier local solution can beat a cleaner change that ripples across four phases.

### Failure Modes

**Roughly 5× the chunk count you expected.** Correct. Each parent plus its four-ish children are all
emitted. Phase 3 indexes everything; Phase 4 searches children only.

**Every parent has exactly one child.** `parent_tokens` is too close to `child_tokens` — the
constructor guards the degenerate case, but 1600/1400 is technically valid and equally pointless. Keep
roughly a 4:1 ratio.

**Retrieval returns parents alongside children.** Phase 4's filter is missing. Until then, searches
will surface both levels and results will look duplicated.

---

## 9. File 8 — `src/ingestion/discovery.py`

### The Problem

`os.listdir()` on a directory containing hundreds of thousands of files returns a list of hundreds of
thousands of strings. `glob.glob("**/*.txt", recursive=True)` across the whole corpus builds the entire
match list in memory before returning anything — so nothing gets processed until the walk finishes, and
you cannot tell whether the program is working or hung.

You need paths delivered one at a time, starting immediately.

### Design Decision

**A generator, using `os.scandir` rather than `os.listdir`.** `scandir` yields `DirEntry` objects that
carry cached `stat` information, so checking whether an entry is a file needs no extra syscall. On large
directories this is several times faster than `listdir` plus `os.path.isfile`.

**Deterministic ordering.** Directories and files are sorted. Filesystem order is arbitrary, and
arbitrary order makes checkpoint-resume unreliable and bugs unreproducible. Sorting a single
directory's entries is cheap; sorting the whole corpus would not be.

**Year filtering pushed into the walk.** Skipping the `2019/` directory should cost nothing, not
require enumerating its contents.

```python
import os
from collections.abc import Iterator
from pathlib import Path

from src.core.exceptions import IngestionError
from src.core.logging import logger


def discover_documents(
    root: Path,
    extensions: set[str] | None = None,
    years: set[str] | None = None,
    limit: int | None = None,
) -> Iterator[Path]:
    """Yield document paths beneath `root`, one at a time, in deterministic order.

    A generator by design: the corpus is 37 GB across ~650k files, and any function
    that returns a list of them stalls for minutes and wastes hundreds of MB before
    a single document is processed.

    Args:
        root: The `contracts/` directory, partitioned by year.
        extensions: Lowercase suffixes to accept, e.g. {".txt"}. None accepts all.
        years: Directory names to include. None walks everything.
        limit: Stop after this many paths. Used by SAMPLE_MODE.

    Raises:
        IngestionError: `root` does not exist.
    """
    if not root.is_dir():
        raise IngestionError(
            "Corpus root not found",
            details={"path": str(root), "hint": "check DATASET_PATH in .env"},
        )

    yielded = 0
    for year_dir in _sorted_subdirs(root):
        if years and year_dir.name not in years:
            continue

        for path in _walk_files(year_dir, extensions):
            yield path
            yielded += 1
            if limit is not None and yielded >= limit:
                logger.info("Discovery limit reached", extra={"limit": limit})
                return

    logger.info("Discovery complete", extra={"documents_found": yielded})


def _sorted_subdirs(root: Path) -> list[Path]:
    """Immediate subdirectories, sorted. Falls back to `root` itself if it has none."""
    with os.scandir(root) as entries:
        dirs = sorted(
            (Path(e.path) for e in entries if e.is_dir(follow_symlinks=False)),
            key=lambda p: p.name,
        )
    # A flat corpus (no year partitioning) still needs to be walked.
    return dirs or [root]


def _walk_files(directory: Path, extensions: set[str] | None) -> Iterator[Path]:
    """Recursively yield files under `directory`, depth-first, sorted at each level."""
    try:
        with os.scandir(directory) as entries:
            listing = sorted(entries, key=lambda e: e.name)
    except (PermissionError, OSError) as exc:
        # One unreadable directory must not end the walk.
        logger.warning(
            "Skipping unreadable directory",
            extra={"path": str(directory), "error": str(exc)},
        )
        return

    subdirs: list[Path] = []
    for entry in listing:
        if entry.is_dir(follow_symlinks=False):
            subdirs.append(Path(entry.path))
            continue
        if not entry.is_file(follow_symlinks=False):
            continue
        path = Path(entry.path)
        if extensions and path.suffix.lower() not in extensions:
            continue
        yield path

    for subdir in subdirs:
        yield from _walk_files(subdir, extensions)


def count_documents(root: Path, extensions: set[str] | None = None) -> int:
    """Count matching files. Walks the whole tree — call once, cache the answer.

    Needed for an accurate progress bar, but on 650k files it takes minutes, so
    the pipeline only calls it when a total is actually wanted.
    """
    return sum(1 for _ in discover_documents(root, extensions))
```

### The Theory: generators, concretely

The distinction is the difference between a pipeline that starts instantly and one that appears broken.

```python
# ❌ Builds the entire list first. Nothing happens for minutes;
#    memory climbs by hundreds of MB.
def find_all(root):
    results = []
    for ...:
        results.append(path)
    return results

for path in find_all(root):    # blocks until the walk is 100% complete
    process(path)

# ✅ Yields as it finds. First document processes in milliseconds;
#    memory stays flat regardless of corpus size.
def find_all(root):
    for ...:
        yield path

for path in find_all(root):    # interleaved with the walk
    process(path)
```

A function containing `yield` does not run when called — it returns a generator object. The body
executes only as you consume it, pausing at each `yield` and resuming on the next iteration. So at any
moment exactly one path exists in memory rather than 650,000.

`yield from _walk_files(subdir, extensions)` is the recursive form: it delegates to a sub-generator and
passes its values straight through, which is how the recursion stays lazy. Writing
`for p in _walk_files(...): yield p` is equivalent but noisier.

### Failure Modes

**Discovery seems to hang.** Distinguish two cases. If nothing has been *yielded*, check the root path
resolves — `print(settings.contracts_dir)`. If documents are flowing but slowly, that is a cold
filesystem cache on 650k files and it is normal on the first pass.

**`RecursionError` on deeply nested trees.** `_walk_files` recurses per directory level. EDGAR is two
levels deep so this cannot happen there; if you point it at something pathological, convert to an
explicit stack.

**Fewer documents than expected.** Check `extensions` — uppercase `.TXT` is handled by the `.lower()`
call, but an unexpected suffix like `.txt.gz` is not.

---

## 10. File 9 — `src/ingestion/checkpoint.py`

### The Problem

A full run takes many hours. It will be interrupted — your laptop sleeps, you press Ctrl+C, the process
is killed. Without persistence, every interruption means starting over, so the run never completes.

The requirement is finer than "remember where I stopped", because file order can change between runs
and documents get edited. What is needed is: for any given document, is it already indexed, and is what
is indexed still current?

### Design Decision

**Record a hash per document, not a position.** A cursor ("stopped at file 480,000") breaks the moment
discovery order changes. A set of `doc_id → content_hash` answers both questions — presence means done,
and a hash mismatch means changed and needing re-ingestion.

**JSON Lines, appended.** Not a single JSON object, because rewriting a 650k-entry file after every
document is quadratic. Not SQLite, because that adds a dependency and locking concerns across
multiprocessing workers for what is fundamentally an append-only log. JSONL appends in constant time
and tolerates truncation — a partially written final line is discarded on load, which matters when the
process was killed mid-write.

**Flush on an interval, not every record.** `flush()` per document means a syscall per document.
Batching by count and time bounds the loss to a few seconds of work.

```python
import json
import os
import time
from pathlib import Path
from types import TracebackType

from src.core.logging import logger


class IngestionCheckpoint:
    """Append-only record of successfully ingested documents, for resumable runs.

    Format: one JSON object per line — {"doc_id", "content_hash", "chunks", "ts"}.
    Append-only so recording a document is O(1) rather than rewriting the file, and
    so a process killed mid-write loses at most the final line.

    Usage:
        with IngestionCheckpoint(settings.checkpoint_dir) as cp:
            if cp.should_skip(doc_id, content_hash):
                continue
            ...
            cp.record(doc_id, content_hash, len(chunks))
    """

    def __init__(
        self,
        directory: Path,
        name: str = "ingestion",
        flush_every: int = 100,
        flush_seconds: float = 10.0,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{name}.jsonl"
        self.flush_every = flush_every
        self.flush_seconds = flush_seconds

        self._done: dict[str, str] = {}
        self._handle = None
        self._pending = 0
        self._last_flush = time.monotonic()

        self._load()

    # ─── context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "IngestionCheckpoint":
        self._handle = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ─── API ───────────────────────────────────────────────────────────────

    def should_skip(self, doc_id: str, content_hash: str) -> bool:
        """True if this exact content was already ingested successfully."""
        return self._done.get(doc_id) == content_hash

    def is_known(self, doc_id: str) -> bool:
        """True if the document was ingested at all, regardless of version.

        A known document with a *different* hash has changed, so Phase 3 must
        `delete_by_doc_id` before upserting or the old chunks linger as ghosts.
        """
        return doc_id in self._done

    def record(self, doc_id: str, content_hash: str, chunk_count: int) -> None:
        if self._handle is None:
            raise RuntimeError("Checkpoint used outside its context manager")

        self._handle.write(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "content_hash": content_hash,
                    "chunks": chunk_count,
                    "ts": time.time(),
                }
            )
            + "\n"
        )
        self._done[doc_id] = content_hash
        self._pending += 1

        elapsed = time.monotonic() - self._last_flush
        if self._pending >= self.flush_every or elapsed >= self.flush_seconds:
            self.flush()

    def flush(self) -> None:
        """Force buffered records to disk.

        `flush()` empties Python's buffer into the OS; `os.fsync` forces the OS to
        write to the physical device. Without fsync, a power loss can lose records
        that Python believes are written.
        """
        if self._handle is None or self._pending == 0:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._pending = 0
        self._last_flush = time.monotonic()

    def close(self) -> None:
        if self._handle is not None:
            self.flush()
            self._handle.close()
            self._handle = None

    def reset(self) -> None:
        """Delete all checkpoint state, forcing a full re-ingestion."""
        self.close()
        self.path.unlink(missing_ok=True)
        self._done.clear()
        logger.warning("Checkpoint reset", extra={"path": str(self.path)})

    @property
    def completed_count(self) -> int:
        return len(self._done)

    # ─── internals ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Rebuild in-memory state from the log, tolerating a truncated tail."""
        if not self.path.exists():
            return

        loaded = 0
        corrupt = 0
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._done[record["doc_id"]] = record["content_hash"]
                    loaded += 1
                except (json.JSONDecodeError, KeyError):
                    # Expected on the last line if a previous run was killed
                    # mid-write. Discard and carry on.
                    corrupt += 1

        logger.info(
            "Checkpoint loaded",
            extra={"documents": loaded, "skipped_corrupt_lines": corrupt},
        )
```

### Why in-memory `dict` and not a database query

650,000 entries of `doc_id` (36 chars) plus `content_hash` (64 chars) is roughly 100 bytes each, so
about 65 MB of dictionary. That is a completely acceptable trade for O(1) skip checks against a disk
query per document.

If the corpus grew to tens of millions, this would not hold and SQLite with an index on `doc_id` would
become correct. Know the limit of the design you chose — that is the difference between a decision and
a habit.

### Failure Modes

**Resume re-processes everything.** Either the checkpoint file is not where you think (print
`cp.path`), or `doc_id` is unstable between runs. `make_doc_id` derives from the path, so passing an
absolute path in one run and a relative path in another yields different IDs. Discovery always yields
paths rooted at `settings.contracts_dir`, which is `.resolve()`d, so this stays consistent — but it is
worth knowing where the fragility is.

**`RuntimeError: Checkpoint used outside its context manager`** — you called `record()` without the
`with` block. The write handle only opens in `__enter__`.

**The file grows unboundedly on repeated runs.** By design — it is an append-only log, so re-ingesting
a changed document appends rather than replaces. `_load` keeps the last value per `doc_id`, so
correctness holds. Compact it by deleting and re-running if it ever becomes unwieldy.

---

## 11. File 10 — `src/ingestion/dead_letter.py`

### The Problem

At 650k documents, thousands will fail. Two wrong responses are tempting and both are bad.

Crashing on the first failure means the run never finishes.

Silently skipping — `except Exception: continue`, which is what the prototype does — means you have no
idea how much of your corpus is missing or why. If 40,000 documents failed to decode, your retrieval
quality is quietly capped and nothing tells you.

### Design Decision

**A dead-letter queue: record the failure with enough context to diagnose and reprocess, then
continue.** The term comes from message-queue systems, where a message that cannot be processed is
routed to a dead-letter queue rather than dropped or retried forever.

**Record the error class, not just the message.** Aggregating by exception type is what turns 3,250
failures into the actionable sentence *"3,100 of these are `Binary content in a text file`, all under
`2001/`"*.

**Copy the offending file only optionally.** Copying every failure could mean gigabytes. Default to
recording the path; enable copying while actively debugging.

```python
import json
import shutil
import time
from collections import Counter
from pathlib import Path

from src.core.exceptions import RAGException
from src.core.logging import logger
from src.core.utils import safe_filename


class DeadLetterQueue:
    """Quarantine for documents that could not be ingested.

    Failures are recorded and the run continues. At corpus scale a pipeline that
    stops on bad input never finishes, and one that skips silently leaves you
    unable to say how much of the corpus is actually indexed.
    """

    def __init__(self, directory: Path, copy_files: bool = False) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log_path = directory / "failures.jsonl"
        self.copy_files = copy_files
        if copy_files:
            (self.directory / "files").mkdir(exist_ok=True)

        self._counts: Counter[str] = Counter()

    def record(self, file_path: str | Path, error: Exception, stage: str) -> None:
        """Log one failure. Never raises — a failing error handler is unforgivable."""
        path = Path(file_path)
        error_type = type(error).__name__

        entry = {
            "path": path.as_posix(),
            "file_name": path.name,
            "stage": stage,
            "error_type": error_type,
            "message": str(error),
            "ts": time.time(),
        }
        if isinstance(error, RAGException):
            entry["details"] = error.details

        try:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, default=str) + "\n")

            if self.copy_files and path.is_file():
                target = self.directory / "files" / safe_filename(f"{stage}_{path.name}")
                shutil.copy2(path, target)
        except Exception as exc:
            # The DLQ itself failing must not take down the pipeline.
            logger.error(
                "Dead-letter write failed",
                extra={"path": str(path), "error": str(exc)},
            )

        self._counts[error_type] += 1
        logger.debug(
            "Quarantined document",
            extra={"path": str(path), "stage": stage, "error_type": error_type},
        )

    def merge_counts(self, counts: dict[str, int]) -> None:
        """Fold in tallies produced by a worker process.

        Workers cannot share this object across the process boundary, so each keeps
        its own counts and the parent merges them at the end.
        """
        self._counts.update(counts)

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def summary(self) -> dict[str, int]:
        return dict(self._counts.most_common())

    def report(self, total_documents: int) -> None:
        """Log an aggregate breakdown. Call once at the end of a run."""
        if not self._counts:
            logger.info("No documents were quarantined.")
            return

        rate = (self.total / total_documents * 100) if total_documents else 0.0
        logger.warning(
            f"Quarantined {self.total} of {total_documents} documents ({rate:.2f}%)",
            extra={"failure_rate_pct": round(rate, 2), "breakdown": self.summary()},
        )
        for error_type, count in self._counts.most_common(10):
            logger.warning(f"  {count:>7,}  {error_type}")

        if rate > 5.0:
            logger.error(
                "Failure rate above 5% — investigate before trusting retrieval "
                "quality, since a large slice of the corpus is not indexed."
            )
```

### What to do with the queue afterwards

The queue is not an archive; it is a work list. The realistic loop is: run ingestion, read the report,
find that most failures share one cause, fix that cause, re-run. Deterministic IDs and the checkpoint
mean the re-run only touches what actually failed.

The 5% threshold is a judgement call worth keeping. Below it you are looking at genuinely corrupt files
that any pipeline would reject. Above it, something systematic is wrong with your loaders — and it is
much better to learn that from a warning than from mysteriously poor answers three phases later.

### Failure Modes

**`failures.jsonl` is empty after a run with obvious problems.** Failures are being caught somewhere
else first. Search for bare `except` blocks in your pipeline.

**The directory grows to gigabytes.** `copy_files=True` was left on. It is a debugging aid.

---

## 12. File 11 — `src/ingestion/pipeline.py`

### The Problem

Now assemble everything, and confront the parallelism question honestly.

Processing one document means: load (I/O), parse (CPU), chunk (CPU, tokenizing is not cheap), and later
embed (CPU, heavily). Serially at ~50ms per document, 650,000 documents is **about nine hours** — and
that is before embedding, which dominates everything else.

### Design Decision — and why `asyncio` is the wrong tool here

This is the most important conceptual point in the phase, because it contradicts the rest of the
project.

`asyncio` provides **concurrency**, not **parallelism**. It lets one thread juggle many operations that
are *waiting* — on a network response, on a disk read. While one coroutine awaits, another runs. That
is perfect for Phase 4's three simultaneous Qdrant queries.

It does nothing for computation. Parsing and tokenizing do not wait; they compute. Python's **Global
Interpreter Lock** permits only one thread to execute bytecode at a time, so neither `asyncio` nor
`threading` will use your second core for CPU work. Wrapping the chunker in `async def` would make it
no faster and merely look concurrent.

**`multiprocessing` is the answer.** Separate processes have separate interpreters and separate GILs,
so eight worker processes genuinely use eight cores.

The consequences shape the whole file:

**Workers must be picklable.** Arguments and return values cross the process boundary via `pickle`.
Plain paths in, Pydantic models out — both pickle fine. A Qdrant client would not.

**Workers do CPU work only; the parent does I/O.** Workers load, parse, and chunk. They return `Chunk`
lists. The parent handles storage, since a database connection per worker is wasteful and awkward.

**Global state must initialise per worker.** Each process needs its own tokenizer. `@lru_cache` on
`get_tokenizer` gives us this for free — the cache is per-process, so each worker loads once.

**On Windows, the guard is mandatory.** Windows lacks `fork`, so it *spawns* a fresh interpreter that
re-imports your module. Without `if __name__ == "__main__":`, that re-import starts the pool again,
recursively, until the machine gives up. This is not a style preference; it is a hard requirement, and
you are on Windows.

```python
import multiprocessing as mp
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import settings
from src.core.exceptions import RAGException
from src.core.logging import logger, new_request_id
from src.core.models import Chunk
from src.core.telemetry import telemetry

from .checkpoint import IngestionCheckpoint
from .chunkers.parent_child_chunker import ParentChildChunker
from .dead_letter import DeadLetterQueue
from .discovery import discover_documents
from .loaders.factory import LoaderFactory
from .parsers.legal_parser import LegalSectionParser
from .parsers.metadata_extractor import ContractMetadataExtractor


@dataclass
class DocumentResult:
    """What a worker returns. Must be picklable — hence plain data only."""

    path: str
    doc_id: str = ""
    content_hash: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    error_type: str | None = None
    error_message: str = ""
    stage: str = ""

    @property
    def ok(self) -> bool:
        return self.error_type is None


@dataclass
class IngestionStats:
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    chunks_produced: int = 0

    def summary(self) -> dict[str, int]:
        return {
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
            "chunks": self.chunks_produced,
        }


# ─── Worker-process globals ────────────────────────────────────────────────
# Built once per worker by `_init_worker`, not per document. Constructing the
# parser and chunker 650,000 times would dwarf the actual work.

_loader: LoaderFactory | None = None
_parser: LegalSectionParser | None = None
_enricher: ContractMetadataExtractor | None = None
_chunker: ParentChildChunker | None = None


def _init_worker() -> None:
    """Initialise per-process state. Runs once in each worker at pool startup."""
    global _loader, _parser, _enricher, _chunker
    _loader = LoaderFactory()
    _parser = LegalSectionParser()
    _enricher = ContractMetadataExtractor()
    _chunker = ParentChildChunker()


def _process_one(path_str: str) -> DocumentResult:
    """Load, parse, and chunk a single document. Runs inside a worker process.

    Must never raise: an exception here would kill the worker and, depending on the
    pool, stall the run. Every failure is converted into a DocumentResult carrying
    the error, which the parent routes to the dead-letter queue.
    """
    assert _loader and _parser and _enricher and _chunker, "worker not initialised"

    stage = "load"
    try:
        document = _loader.load(path_str)
        stage = "enrich"
        document = _enricher.enrich(document)

        stage = "parse"
        sections = _parser.parse(document)

        stage = "chunk"
        chunks: list[Chunk] = []
        for section in sections:
            chunks.extend(_chunker.chunk(section, document))

        if not chunks:
            return DocumentResult(
                path=path_str,
                error_type="EmptyChunkSet",
                error_message="Document produced no chunks",
                stage="chunk",
            )

        return DocumentResult(
            path=path_str,
            doc_id=document.doc_id,
            content_hash=document.content_hash,
            chunks=chunks,
        )

    except RAGException as exc:
        return DocumentResult(
            path=path_str, error_type=type(exc).__name__,
            error_message=str(exc), stage=stage,
        )
    except Exception as exc:
        # Genuinely unexpected. Still must not escape the worker.
        return DocumentResult(
            path=path_str, error_type=type(exc).__name__,
            error_message=f"Unhandled: {exc}", stage=stage,
        )


class IngestionPipeline:
    """Pipeline 1 orchestrator: discovery → load → parse → chunk.

    Produces `Chunk` objects and stops. Storing them is Phase 3's responsibility,
    which is why `run()` yields batches rather than writing anywhere.
    """

    def __init__(
        self,
        workers: int | None = None,
        batch_size: int = 256,
        copy_failed_files: bool = False,
    ) -> None:
        self.workers = workers or max(1, (mp.cpu_count() or 2) - 1)
        self.batch_size = batch_size
        self.stats = IngestionStats()
        self.dlq = DeadLetterQueue(settings.dead_letter_dir, copy_files=copy_failed_files)

    @telemetry.span("ingestion.run")
    def run(
        self,
        limit: int | None = None,
        years: set[str] | None = None,
        resume: bool = True,
    ) -> Iterator[list[Chunk]]:
        """Process the corpus, yielding batches of chunks.

        A generator so the caller can stream batches into the vector store instead
        of accumulating every chunk in memory — the whole corpus would be tens of
        millions of chunks.

        Args:
            limit: Maximum documents to process. Defaults to SAMPLE_LIMIT when
                SAMPLE_MODE is on.
            years: Restrict to these year directories.
            resume: Skip documents already recorded in the checkpoint.
        """
        new_request_id()

        if limit is None and settings.SAMPLE_MODE:
            limit = settings.SAMPLE_LIMIT
            logger.warning(
                "SAMPLE_MODE is on — processing a subset only",
                extra={"limit": limit},
            )

        extensions = LoaderFactory().supported_extensions
        paths = discover_documents(
            settings.contracts_dir, extensions=extensions, years=years, limit=limit
        )

        logger.info(
            "Starting ingestion",
            extra={
                "workers": self.workers,
                "batch_size": self.batch_size,
                "resume": resume,
                "limit": limit,
            },
        )

        with IngestionCheckpoint(settings.checkpoint_dir) as checkpoint:
            if not resume:
                checkpoint.reset()
            elif checkpoint.completed_count:
                logger.info(
                    "Resuming from checkpoint",
                    extra={"already_done": checkpoint.completed_count},
                )

            yield from self._run_pool(paths, checkpoint)

        self.dlq.report(self.stats.processed + self.stats.failed)
        logger.info("Ingestion complete", extra=self.stats.summary())

    # ─── internals ─────────────────────────────────────────────────────────

    def _run_pool(
        self, paths: Iterator[Path], checkpoint: IngestionCheckpoint
    ) -> Iterator[list[Chunk]]:
        batch: list[Chunk] = []
        worker_error_counts: Counter[str] = Counter()

        # `imap_unordered` over `map`: it streams results as they finish instead of
        # materialising the full result list, and it does not wait for slow documents
        # to preserve ordering — ordering is irrelevant here.
        #
        # `chunksize` batches task dispatch. Sending 650k paths one at a time makes
        # IPC overhead dominate; 32 amortises it without starving workers.
        with mp.Pool(processes=self.workers, initializer=_init_worker) as pool:
            results = pool.imap_unordered(
                _process_one, (str(p) for p in paths), chunksize=32
            )

            for result in results:
                if not result.ok:
                    self.stats.failed += 1
                    worker_error_counts[result.error_type or "Unknown"] += 1
                    self.dlq.record(
                        result.path,
                        RuntimeError(result.error_message),
                        stage=result.stage,
                    )
                    continue

                if checkpoint.should_skip(result.doc_id, result.content_hash):
                    self.stats.skipped += 1
                    continue

                batch.extend(result.chunks)
                self.stats.processed += 1
                self.stats.chunks_produced += len(result.chunks)
                checkpoint.record(result.doc_id, result.content_hash, len(result.chunks))

                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []

                if self.stats.processed % 1000 == 0:
                    logger.info("Progress", extra=self.stats.summary())

        if batch:
            yield batch

        self.dlq.merge_counts(dict(worker_error_counts))
```

### Why skipping happens in the parent, not the worker

Look closely: the worker fully processes a document, and only then does the parent check
`should_skip`. That looks wasteful — why not skip before doing the work?

Because the skip decision needs `content_hash`, and computing that requires loading and normalising the
document. The check is "has this *content* been ingested", not "has this path been seen". Path-based
skipping would miss edited documents entirely.

There is a cheaper pre-filter available and it is worth knowing about: `hash_file` on the raw bytes
needs no decoding or parsing, so a two-tier scheme — byte hash to skip in the parent before dispatch,
content hash to confirm — would avoid most of the wasted work on resume. We do not implement it here
because it adds a second hash store for a cost that only appears on repeated runs of an already-complete
corpus. Worth adding if you find yourself resuming often.

### Failure Modes

**The process spawns endlessly and the machine locks up (Windows).** You omitted
`if __name__ == "__main__":` in the entry-point script. Non-negotiable on Windows:

```python
if __name__ == "__main__":
    pipeline = IngestionPipeline()
    for batch in pipeline.run(limit=100):
        print(len(batch))
```

**`PicklingError: Can't pickle <class '...'>`** — something unpicklable is crossing the boundary. Usual
suspects are a lambda, a local function, an open file handle, or a database client. Keep worker
arguments to primitives and returns to Pydantic models.

**Workers idle while one core saturates.** `chunksize` too large with uneven document sizes, so one
worker got a batch of enormous files. Lower it.

**Memory climbs steadily through the run.** Something is accumulating. `run()` yields batches and clears
`batch`; make sure your *caller* is not appending every batch to a list. The generator exists precisely
so it does not have to.

**Slower with 8 workers than with 1** on a small corpus. Process startup costs a few hundred
milliseconds each. Below a few thousand documents, that dominates. Use `workers=1` for testing.

---

## 13. Verification (deferred)

Save as `scripts/verify_phase2.py`. Run on the machine with the corpus, after Phase 1's script passes.

```python
"""Phase 2 verification. Run from the project root:
       python scripts/verify_phase2.py
Requires a few sample .txt files under $DATASET_PATH/contracts/<year>/.
"""

from pathlib import Path

from config.settings import settings
from src.core.logging import logger, new_request_id
from src.core.models import ChunkLevel
from src.core.utils import count_tokens
from src.ingestion.chunkers.parent_child_chunker import ParentChildChunker
from src.ingestion.chunkers.sentence_chunker import SentenceChunker, split_sentences
from src.ingestion.discovery import discover_documents
from src.ingestion.loaders.factory import LoaderFactory
from src.ingestion.parsers.legal_parser import LegalSectionParser
from src.ingestion.parsers.metadata_extractor import ContractMetadataExtractor
from src.ingestion.pipeline import IngestionPipeline


def check_sentence_splitting() -> None:
    """The abbreviation cases that break a naive `text.split(". ")`."""
    text = (
        "Acme Inc. shall indemnify Beta Corp. against all claims. "
        "See SECTION 7.02 for details. The cap is $2,000,000. "
        "Notice under Art. 4 must be given to J. Smith at 10 Main St. before closing."
    )
    sentences = split_sentences(text)
    for sentence in sentences:
        logger.info(f"  SENTENCE: {sentence}")

    joined = " ".join(sentences)
    for abbreviation in ("Inc.", "Corp.", "Art.", "J.", "St."):
        assert abbreviation in joined, f"lost {abbreviation} during splitting"

    # Naive splitting on ". " yields 8+ fragments here. We expect 4.
    assert len(sentences) == 4, f"expected 4 sentences, got {len(sentences)}: {sentences}"
    logger.info("Sentence splitting handles legal abbreviations")


def check_chunker_terminates() -> None:
    """Guard against the infinite-loop failure mode in the overlap calculation."""
    text = " ".join(f"This is sentence number {i} of the test paragraph." for i in range(200))

    # A pathological config: overlap larger than the chunk can hold.
    chunker = SentenceChunker(max_tokens=60, overlap_sentences=10)
    chunks = chunker.split_text(text)
    assert chunks, "chunker produced nothing"
    logger.info(f"Chunker terminated under adversarial overlap — {len(chunks)} chunks")

    normal = SentenceChunker(max_tokens=400, overlap_sentences=2)
    produced = normal.split_text(text)
    over_budget = [c for c in produced if count_tokens(c) > 400 * 1.1]
    assert not over_budget, f"{len(over_budget)} chunks exceeded the token budget"
    logger.info(f"Token budget respected across {len(produced)} chunks")


def check_end_to_end(sample_path: Path) -> None:
    loader = LoaderFactory()
    document = ContractMetadataExtractor().enrich(loader.load(str(sample_path)))
    logger.info(f"Loaded: {document.file_name} ({document.char_count:,} chars)")
    logger.info(f"  title : {document.contract_name}")
    logger.info(f"  year  : {document.year}")
    logger.info(f"  parties: {document.metadata.get('parties', [])}")

    sections = LegalSectionParser().parse(document)
    logger.info(f"Parsed {len(sections)} sections")
    for section in sections[:5]:
        logger.info(f"  [{section.order}] {section.title}  ({section.char_count:,} chars)")

    if len(sections) == 1 and sections[0].title == "Preamble":
        logger.warning(
            "Only a Preamble section was found. Either this document is unstructured, "
            "or normalise_whitespace is stripping newlines — check its final regex "
            "is [ \\t\\r\\f\\v]+ and not \\s+."
        )

    chunks: list[Chunk] = []
    chunker = ParentChildChunker()
    for section in sections:
        chunks.extend(chunker.chunk(section, document))

    parents = [c for c in chunks if c.chunk_level is ChunkLevel.PARENT]
    children = [c for c in chunks if c.chunk_level is ChunkLevel.CHILD]
    logger.info(f"Chunked into {len(parents)} parents and {len(children)} children")

    assert children, "no child chunks produced"
    assert all(c.parent_id for c in children), "a child chunk has no parent_id"
    parent_ids = {p.chunk_id for p in parents}
    orphans = [c for c in children if c.parent_id not in parent_ids]
    assert not orphans, f"{len(orphans)} children reference a missing parent"
    logger.info("Parent-child links are intact")

    # IDs must be unique, or chunks silently overwrite each other in Qdrant.
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk_id — parents and children collided"
    logger.info("All chunk IDs unique")

    if children:
        sample = children[0]
        logger.info(f"Example child ({sample.token_count} tokens): {sample.text[:200]}...")


def check_pipeline() -> None:
    """Small parallel run. Confirms the multiprocessing path works end to end."""
    pipeline = IngestionPipeline(workers=2, batch_size=64)
    total = 0
    batches = 0
    for batch in pipeline.run(limit=20, resume=False):
        total += len(batch)
        batches += 1
    logger.info(f"Pipeline produced {total} chunks in {batches} batches")
    logger.info(f"Stats: {pipeline.stats.summary()}")
    logger.info(f"Quarantined: {pipeline.dlq.summary()}")
    assert total > 0, "pipeline produced no chunks at all"


def main() -> None:
    new_request_id()
    logger.info("Phase 2 verification starting")

    check_sentence_splitting()
    check_chunker_terminates()

    samples = list(
        discover_documents(settings.contracts_dir, extensions={".txt"}, limit=1)
    )
    if not samples:
        logger.error(
            f"No .txt files under {settings.contracts_dir} — "
            "set DATASET_PATH and place a few sample contracts before continuing."
        )
        return

    check_end_to_end(samples[0])
    check_pipeline()

    logger.info("PHASE 2 VERIFIED — ingestion produces well-formed chunks.")


if __name__ == "__main__":
    # Mandatory on Windows: without this guard, multiprocessing re-imports this
    # module in each spawned worker and recursively spawns more pools.
    main()
```

### What to look at, beyond the assertions

**Read a few chunks out loud.** This is the highest-value five minutes in the phase and no assertion
replaces it. Does each chunk read as a coherent, self-contained statement? Does any end mid-clause? If
chunks look wrong to you, retrieval will be wrong too, and no amount of reranking downstream will
repair it. Chunk quality sets the ceiling on the entire system.

**Check the parent-to-child ratio.** Roughly 1:4 is expected with 1600/400 tokens. Near 1:1 means the
sizes are too close and the hierarchy is doing nothing.

**Read the dead-letter breakdown.** On a small sample it should be nearly empty. If a quarter of your
sample is quarantined, fix that before scaling up — the failure will not get proportionally better at
650k documents.

---

## 14. What Phase 2 Bought You

You now have Pipeline 1 in full: a memory-flat generator walk over a 37 GB corpus, encoding-robust
loading, structure-aware section parsing that makes clause-level citation possible, sentence-aligned
chunking that does not cut through clauses, a two-level hierarchy that decouples retrieval granularity
from generation context, resumable checkpointing, and a quarantine that turns thousands of failures
into a diagnosable report.

The three ideas worth carrying forward are: **stream, never accumulate** — the generator discipline is
what makes corpus scale tractable at all; **`asyncio` for waiting, `multiprocessing` for computing** —
confusing the two is the most common performance mistake in Python; and **quarantine, never crash and
never silently skip** — at scale, failure is a category to be measured rather than an event to be
prevented.

**Next:** `Phase3_VectorStores_Embeddings.md` — dense and sparse embedding providers behind the async
interface, and the Qdrant adapter using the modern `query_points` API with named vectors, payload
indexes on `doc_id` and `chunk_level`, and the mandatory `delete_by_doc_id` before upsert.
