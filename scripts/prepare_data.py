"""Build the data splits and write the small result files to results/."""

from pathlib import Path

from hiring_bias_xai import data
from hiring_bias_xai.config import RESULTS_DIR


def main(results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    subset = data.xai_subset()
    ids = "".join(f"{bio_id}\n" for bio_id in subset["id"])
    (results_dir / "xai_subset_ids.csv").write_bytes(f"id\n{ids}".encode())


if __name__ == "__main__":
    main()
