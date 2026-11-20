"""Download the SSA baby-names file (pinned sha256), keep the first names with at least
`first_names.MIN_COUNT` births, and write them to `HBX_ARTIFACTS/names/first_names.txt`.
`results/first_names.json` records the source, hash, threshold and number of names.
"""

from pathlib import Path

from hiring_bias_xai import first_names, results
from hiring_bias_xai.config import RESULTS_DIR, artifacts_dir


def main(results_dir: Path = RESULTS_DIR) -> None:
    archive = first_names.download(
        first_names.URL, first_names.SHA256, artifacts_dir() / "names" / "ssa_names.zip"
    )
    names = first_names.parse(archive)
    first_names.names_file().write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    results.write_json(
        results_dir / "first_names.json",
        {
            "url": first_names.URL,
            "sha256": first_names.SHA256,
            "min_count": first_names.MIN_COUNT,
            "n_names": len(names),
        },
    )


if __name__ == "__main__":
    main()
