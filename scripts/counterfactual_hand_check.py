"""Draw the bios for the hand check of the gender swap and print their changed sentences.

`python scripts/counterfactual_hand_check.py` draws 50 bios at random (`config.SEED`) among those
of the showcase subset that the swap changes, and prints, for each, only the sentences that
differ, before and after. The verdicts go by hand into `results/counterfactual_hand_check.csv`
(columns id, coherent yes/no, reason for every no); `scripts/counterfactual_gender.py` reads it.
"""

import re
from pathlib import Path

from hiring_bias_xai import counterfactual, counterfactual_effects, data
from hiring_bias_xai.config import RESULTS_DIR

N_BIOS = 50
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def changed_sentences(original: str, swapped: str) -> list[tuple[str, str]]:
    """The (before, after) pairs of sentences that differ; the swap never moves a sentence end."""
    pairs = zip(_SENTENCE_END.split(original), _SENTENCE_END.split(swapped), strict=True)
    return [(before, after) for before, after in pairs if before != after]


def draw(ids_file: Path, n: int = N_BIOS) -> list[tuple[str, list[tuple[str, str]]]]:
    """`n` bio ids drawn from the bios whose text the swap changes, with their changed sentences."""
    subset = data.load_showcase_subset(ids_file)
    swapped = {
        bio.id: counterfactual.swap_gender(bio.text) for bio in subset.itertuples(index=False)
    }
    changed = [bio for bio in subset.itertuples(index=False) if swapped[bio.id] != bio.text]
    texts = {bio.id: bio.text for bio in changed}
    sample = counterfactual_effects.hand_check_sample([bio.id for bio in changed], n)
    return [(bio_id, changed_sentences(texts[bio_id], swapped[bio_id])) for bio_id in sample]


def main(results_dir: Path = RESULTS_DIR) -> None:
    for bio_id, pairs in draw(results_dir / "showcase_subset_ids.csv"):
        print(f"== {bio_id}")
        for before, after in pairs:
            print(f"- {before}\n+ {after}")


if __name__ == "__main__":
    main()
