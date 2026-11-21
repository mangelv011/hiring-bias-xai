"""Check the written documents against the repository: `python scripts/check_docs.py`.

Reads files of this repository only and makes no network calls. Exits with 1 and prints one line
per problem when a link or an image path does not resolve.
"""

import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`[^`\n]*`")
MARKDOWN_LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_REFERENCE = re.compile(r"""\b(?:src|href)\s*=\s*["']([^"']+)["']""")
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
            if resolve(root, document, target) is None:
                errors.append(f"{name}:{number}: {target} does not resolve")
    return errors


def main(root: Path = REPO_ROOT) -> int:
    errors = check_links(root)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
