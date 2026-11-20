"""F3: the words that make up the main model's score gap, in two showcase occupations.

One panel for a male-dominated and one for a female-dominated showcase occupation. The rule,
fixed before looking at the figure: in each kind, the showcase occupation with the largest
absolute score gap of the main model (mean score of women's bios minus men's, `score_gap` in
`results/gap_explained.json`). Each panel shows the ten words with the largest absolute
contribution to that gap, with their 95% bootstrap intervals.
"""

from collections.abc import Mapping, Sequence

KINDS = ("male_dominated", "female_dominated")


def pick(summaries: Mapping[str, dict], showcase: Mapping[str, Sequence[dict]]) -> dict:
    """For each kind of showcase occupation, the one with the largest absolute score gap.

    `summaries` maps occupation to the main model's summary in `gap_explained.json`; a tie goes
    to the occupation listed first.
    """
    return {
        kind: max(
            (row["occupation"] for row in showcase[kind]),
            key=lambda occupation: abs(summaries[occupation]["score_gap"]["value"]),
        )
        for kind in KINDS
    }
