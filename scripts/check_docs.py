"""Check the written documents against the repository: `python scripts/check_docs.py`.

Reads files of this repository only and makes no network calls. Exits with 1 and prints one line
per problem when a link, an image path or a heading anchor does not resolve.
"""

import csv
import json
import os
import re
import sys
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`[^`\n]*`")
MARKDOWN_LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_REFERENCE = re.compile(r"""\b(?:src|href)\s*=\s*["']([^"']+)["']""")
HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
HEADING_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
HTML_TAG = re.compile(r"<[^>]*>")
NOT_IN_SLUG = re.compile(r"[^\w\s-]")
NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
EXTERNAL = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)


def documents(root: Path) -> list[Path]:
    """README.md, the pages of docs/ and the decision records, in a fixed order."""
    found = [root / "README.md"]
    found += sorted((root / "docs").glob("*.md"))
    found += sorted((root / "docs" / "decisions").glob("*.md"))
    return [path for path in found if path.is_file()]


def prose_lines(text: str):
    """Yield (line number, line) for the lines outside fenced code, inline code removed."""
    fenced = False
    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            yield number, INLINE_CODE.sub("", line)


def references(text: str):
    """Yield (line number, target) for every link and image of a document."""
    for number, line in prose_lines(text):
        for pattern in (MARKDOWN_LINK, HTML_REFERENCE):
            for match in pattern.finditer(line):
                yield number, match.group(1)


def slug(heading: str) -> str:
    """GitHub anchor of a heading: no markup or punctuation, lower case, spaces as hyphens."""
    text = HTML_TAG.sub("", HEADING_LINK.sub(r"\1", heading))
    return NOT_IN_SLUG.sub("", text.lower()).replace(" ", "-")


def anchors(text: str) -> set[str]:
    """The anchors of the headings of a document; a repeated heading gets -1, -2 and so on."""
    seen: dict[str, int] = {}
    found = set()
    fenced = False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
            continue
        match = None if fenced else HEADING.match(line)
        if match:
            base = slug(match.group(1))
            count = seen.get(base, 0)
            seen[base] = count + 1
            found.add(base if count == 0 else f"{base}-{count}")
    return found


def exists_with_case(root: Path, relative: Path) -> bool:
    """True when every part of `relative` exists under `root` with exactly this spelling."""
    here = root
    for part in relative.parts:
        if part not in {entry.name for entry in here.iterdir()}:
            return False
        here = here / part
    return True


def resolve(root: Path, document: Path, target: str) -> Path | None:
    """The path a relative target points to, or None when it is missing."""
    path = unquote(target.split("#", 1)[0])
    if not path:
        return document
    base = Path(os.path.normpath(root))
    here = Path(os.path.normpath(document.parent / path))
    try:
        relative = here.relative_to(base)
    except ValueError:
        return None
    return here if exists_with_case(base, relative) else None


def check_links(root: Path) -> list[str]:
    """One message per link or image of the documents whose target does not resolve."""
    errors = []
    for document in documents(root):
        name = document.relative_to(root).as_posix()
        text = document.read_text(encoding="utf-8")
        for number, target in references(text):
            if EXTERNAL.match(target):
                continue
            found = resolve(root, document, target)
            if found is None:
                errors.append(f"{name}:{number}: {target} does not resolve")
                continue
            _, _, fragment = target.partition("#")
            if fragment and found.suffix == ".md":
                if unquote(fragment) not in anchors(found.read_text(encoding="utf-8")):
                    errors.append(f"{name}:{number}: {target} has no such heading")
    return errors


class Cell(NamedTuple):
    """One number a document states and where it comes from in results/.

    `pattern` has one group holding the number as written. `path` is dotted keys for a JSON file
    (a list is entered by position) or `column=value,...:column` for a CSV file.
    """

    document: str
    pattern: str
    file: str
    path: str
    fmt: str


class Allowed(NamedTuple):
    """A phrase whose numbers are not results: a year, a count in a method description."""

    document: str | None
    pattern: str
    reason: str


NUMBER_DOCUMENTS = ("README.md", "docs/model_card.md")
CELLS: list[Cell] = []
ALLOWED: list[Allowed] = []


def _fixed(scale: Decimal, decimals: int, signed: bool = False):
    """A formatter: the value times `scale`, rounded half up to `decimals` places."""
    step = Decimal(1).scaleb(-decimals)

    def fmt(value) -> str:
        rounded = (Decimal(repr(value)) * scale).quantize(step, rounding=ROUND_HALF_UP)
        return f"{rounded:+f}" if signed else f"{rounded:f}"

    return fmt


FORMATS = {
    "per100": _fixed(Decimal(100), 0),
    "pct1": _fixed(Decimal(100), 1),
    "dec1": _fixed(Decimal(1), 1),
    "dec2": _fixed(Decimal(1), 2),
    "dec3": _fixed(Decimal(1), 3),
    "dec4": _fixed(Decimal(1), 4),
    "signed_points1": _fixed(Decimal(100), 1, signed=True),
    "signed3": _fixed(Decimal(1), 3, signed=True),
    "hours2": _fixed(Decimal(1) / Decimal(3600), 2),
    "minutes0": _fixed(Decimal(1) / Decimal(60), 0),
    "count": lambda value: f"{int(value):,}",
}


def read_value(results_dir: Path, file: str, path: str):
    """The value at `path` of a results file; raises when the file, path or row is missing."""
    full = results_dir / file
    if full.suffix == ".csv":
        keys, _, column = path.rpartition(":")
        wanted = dict(item.split("=", 1) for item in keys.split(","))
        with full.open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if all(r[k] == v for k, v in wanted.items())]
        if len(rows) != 1:
            raise ValueError(f"{len(rows)} rows match")
        return float(rows[0][column])
    value = json.loads(full.read_text(encoding="utf-8"))
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _blank(line: str, pattern: re.Pattern) -> str:
    """The line with the first group of every match of `pattern` replaced by spaces."""
    for match in pattern.finditer(line):
        start, end = match.span(1)
        line = line[:start] + " " * (end - start) + line[end:]
    return line


def claim_lines(text: str) -> list[tuple[int, str]]:
    """The lines a reader takes numbers from: no code, no link targets, no image sources."""
    return [
        (number, _blank(_blank(line, MARKDOWN_LINK), HTML_REFERENCE))
        for number, line in prose_lines(text)
    ]


def number_tokens(line: str):
    """Yield (start, end, text) of the numbers of a line, leaving out names such as F1 or 9B."""
    for match in NUMBER.finditer(line):
        before = line[match.start() - 1 : match.start()]
        after = line[match.end() : match.end() + 1]
        if before and (before.isalnum() or before == "_"):
            continue
        if after and (after.isalpha() or after == "_"):
            continue
        yield match.start(), match.end(), match.group()


def _covered(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(a <= start and end <= b for a, b in spans)


def check_numbers(
    root: Path,
    cells: Sequence[Cell] = CELLS,
    allowed: Sequence[Allowed] = ALLOWED,
    documents: Sequence[str] = NUMBER_DOCUMENTS,
) -> list[str]:
    """One message per number that differs from results/ and per number no row or phrase covers."""
    errors = [
        f"{c.document}: a row of the table names a document that is not checked"
        for c in cells
        if c.document not in documents
    ]
    for name in documents:
        path = root / name
        if not path.is_file():
            errors.append(f"{name}: document is missing")
            continue
        lines = claim_lines(path.read_text(encoding="utf-8"))
        spans: dict[int, list[tuple[int, int]]] = {}
        for cell in (c for c in cells if c.document == name):
            pattern = re.compile(cell.pattern)
            found = [(n, m) for n, line in lines for m in pattern.finditer(line)]
            if not found:
                errors.append(f"{name}: no text matches {cell.pattern!r}")
                continue
            for number, match in found:
                spans.setdefault(number, []).append(match.span(1))
            try:
                expected = FORMATS[cell.fmt](read_value(root / "results", cell.file, cell.path))
            except (OSError, KeyError, ValueError, IndexError) as exc:
                errors.append(f"{name}: cannot read {cell.file} {cell.path}: {exc!r}")
                continue
            for number, match in found:
                if match.group(1) != expected:
                    errors.append(
                        f"{name}:{number}: {match.group(1)} in the text, {expected} in "
                        f"{cell.file} {cell.path}"
                    )
        for entry in (a for a in allowed if a.document in (None, name)):
            pattern = re.compile(entry.pattern)
            for number, line in lines:
                spans.setdefault(number, []).extend(m.span() for m in pattern.finditer(line))
        for number, line in lines:
            for start, end, token in number_tokens(line):
                if not _covered(spans.get(number, []), start, end):
                    errors.append(
                        f"{name}:{number}: number {token} is covered by no row or allowed phrase"
                    )
    return errors


def main(root: Path = REPO_ROOT) -> int:
    errors = check_links(root)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
