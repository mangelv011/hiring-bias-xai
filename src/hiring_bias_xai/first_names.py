"""First names from the US Social Security Administration baby-names file, pinned by sha256.

The archive is data, never code: it is checked against a hash before anything reads it, and
only text members named `yobYYYY.txt` are parsed, in memory, with a cap on their total size.
"""

import hashlib
import io
import re
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from hiring_bias_xai.config import artifacts_dir

# The SSA site refuses automated clients, so the pin is an Internet Archive capture of
# https://www.ssa.gov/oact/babynames/names.zip (`id_` serves the original bytes).
URL = "https://web.archive.org/web/20261102233045id_/https://www.ssa.gov/oact/babynames/names.zip"
SHA256 = "cd78e975ed7bb358e018dd62fbe14ced89295e9581c49172ca4eedcb011b3724"
MIN_COUNT = 10_000  # births over all years and both sexes
MAX_DOWNLOAD_BYTES = 50_000_000
MAX_UNPACKED_BYTES = 100_000_000

_MEMBER = re.compile(r"yob\d{4}\.txt")
_LINE = re.compile(r"([A-Za-z]+),[FM],(\d+)")


def parse(archive: bytes, min_count: int = MIN_COUNT, max_bytes: int = MAX_UNPACKED_BYTES) -> list:
    """The names with at least `min_count` births in the `yobYYYY.txt` files, sorted.

    Each line is `name,sex,count`; counts are summed over years and sexes. Raises ValueError if
    the members unpack to more than `max_bytes` or a line has another shape.
    """
    totals: Counter = Counter()
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        members = [info for info in zipped.infolist() if _MEMBER.fullmatch(info.filename)]
        if sum(info.file_size for info in members) > max_bytes:
            raise ValueError(f"the archive unpacks to more than {max_bytes} bytes: too large")
        for info in members:
            for line in zipped.read(info).decode("utf-8").splitlines():
                parsed = _LINE.fullmatch(line)
                if parsed is None:
                    raise ValueError(f"{info.filename}: unexpected line {line!r}")
                totals[parsed.group(1)] += int(parsed.group(2))
    return sorted(name for name, count in totals.items() if count >= min_count)


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:  # `download` allows https only
        content = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"{url} is larger than {MAX_DOWNLOAD_BYTES} bytes")
    return content


def download(
    url: str, sha256: str, target: Path, fetch: Callable[[str], bytes] = _fetch
) -> bytes:
    """Fetch `url` (https only), check its sha256 and only then write it to `target`.

    Raises ValueError for another scheme or a different hash; nothing is written then.
    """
    if not url.startswith("https://"):
        raise ValueError(f"only https URLs are fetched, not {url}")
    content = fetch(url)
    actual = hashlib.sha256(content).hexdigest()
    if actual != sha256:
        raise ValueError(f"sha256 of {url} is {actual}, expected {sha256}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return content


def names_file() -> Path:
    """Where `scripts/fetch_first_names.py` writes the list, one name per line."""
    return artifacts_dir() / "names" / "first_names.txt"


def load() -> frozenset:
    """The names that `scripts/fetch_first_names.py` wrote, Capitalised."""
    try:
        return frozenset(names_file().read_text(encoding="utf-8").split())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{names_file()} is missing: run scripts/fetch_first_names.py first"
        ) from None
