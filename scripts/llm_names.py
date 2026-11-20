"""Download the four INE name lists (pinned sha256) and write the name pools of the LLM audit.

The files go to `HBX_ARTIFACTS/ine/` and stay out of git. `results/llm/names.json` holds, for
each origin group, the top first names per sex and the frequent surnames, with the source,
dates, licence and attribution of the INE data (CC BY 4.0).
"""

from pathlib import Path

from hiring_bias_xai import first_names, ine_names, results
from hiring_bias_xai.config import RESULTS_DIR, artifacts_dir


def main(results_dir: Path = RESULTS_DIR) -> None:
    sheets = {}
    for key, (url, sha256) in ine_names.FILES.items():
        content = first_names.download(url, sha256, artifacts_dir() / "ine" / Path(url).name)
        sheets[key] = ine_names.read_sheets(content)
    built = ine_names.build_groups(sheets)
    out = results_dir / "llm"
    out.mkdir(parents=True, exist_ok=True)
    results.write_json(
        out / "names.json",
        {
            "source": {
                "publisher": "Instituto Nacional de Estadística (INE)",
                "licence": ine_names.LICENCE,
                "attribution": ine_names.ATTRIBUTION,
                "files": {
                    key: {"url": url, "sha256": sha256, "date": ine_names.DATES[key]}
                    for key, (url, sha256) in ine_names.FILES.items()
                },
            },
            "groups": built["groups"],
            "dropped_surnames": built["dropped_surnames"],
            "n_dropped_surnames": len(built["dropped_surnames"]),
            "shared_surnames": ine_names.shared_surnames(built["groups"]),
        },
    )


if __name__ == "__main__":
    main()
