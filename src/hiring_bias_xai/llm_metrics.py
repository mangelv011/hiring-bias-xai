"""Metrics of the LLM audit, on a table with one row per answered main call.

Columns: `pid` (a profile: one bio for one job), `match`, `arm`, `kind` (none, neutral, swap or
name), `group` and `name_gender` and `first` (the name arms only), `yes` (1, 0 or NaN for an
invalid answer) and `reason`.
"""

import numpy as np
import pandas as pd

from hiring_bias_xai.config import SEED
from hiring_bias_xai.gap_statistics import holm
from hiring_bias_xai.llm_audit import REFERENCE_GROUP

N_BOOT = 1000


def arm_rates(calls: pd.DataFrame) -> dict:
    """Per arm: `tpr` (yes share on matches), `fpr` (yes share on non-matches), the number of
    valid answers on each and `n_invalid`. Invalid answers are in no rate."""
    rates = {}
    for arm, rows in calls.groupby("arm", sort=False):
        valid = rows[rows["yes"].notna()]
        matches, others = valid[valid["match"]], valid[~valid["match"]]
        rates[arm] = {
            "tpr": float(matches["yes"].mean()),
            "fpr": float(others["yes"].mean()),
            "n_match": len(matches),
            "n_nonmatch": len(others),
            "n_invalid": int(rows["yes"].isna().sum()),
        }
    return rates


class _Design:
    """Row indices for the two-level bootstrap: profiles, and first names within each cell
    (origin group and name gender). A row weighs its profile's multiplicity times its first
    name's multiplicity; a row without a name has no second factor."""

    def __init__(self, calls: pd.DataFrame):
        self.calls = calls.reset_index(drop=True)
        self.profile = pd.factorize(self.calls["pid"])[0]
        self.n_profiles = self.profile.max() + 1
        named = (self.calls["kind"] == "name").to_numpy()
        keys = list(
            zip(self.calls["group"], self.calls["name_gender"], self.calls["first"], strict=True)
        )
        self.cluster = np.full(len(self.calls), -1)
        cells: dict = {}
        ids: dict = {}
        for i in np.flatnonzero(named):
            cluster = ids.setdefault(keys[i], len(ids))
            self.cluster[i] = cluster
            cells.setdefault(keys[i][:2], set()).add(cluster)
        self.n_clusters = len(ids)
        self.cells = [np.array(sorted(members)) for members in cells.values()]
        self.yes = self.calls["yes"].to_numpy()
        self.match = self.calls["match"].to_numpy(dtype=bool)
        self.valid = ~np.isnan(self.yes)

    def weights(self, rng: np.random.Generator | None):
        """Row weights and cluster weights of one resample (all ones when `rng` is None)."""
        if rng is None:
            profiles, clusters = np.ones(self.n_profiles), np.ones(self.n_clusters)
        else:
            profiles = rng.multinomial(
                self.n_profiles, np.full(self.n_profiles, 1 / self.n_profiles)
            )
            clusters = np.ones(self.n_clusters)
            for members in self.cells:
                clusters[members] = rng.multinomial(
                    len(members), np.full(len(members), 1 / len(members))
                )
        named = self.cluster >= 0
        rows = profiles[self.profile] * np.where(
            named, clusters[np.where(named, self.cluster, 0)], 1
        )
        return rows.astype(float), clusters, profiles


def _share(design: _Design, weights, rows: np.ndarray, matches: bool) -> float:
    """Weighted yes share over the valid rows of `rows` that are matches (or non-matches)."""
    chosen = rows & design.valid & (design.match == matches)
    total = weights[chosen].sum()
    return float((weights[chosen] * design.yes[chosen]).sum() / total) if total else np.nan


def _statistics(design: _Design, reference: str) -> dict:
    """Every statistic as a function of (row weights, cluster weights, profile weights)."""
    calls = design.calls
    arm = calls["arm"].to_numpy()
    arms = list(dict.fromkeys(arm))
    stats = {}
    for name in arms:
        rows = arm == name
        stats[f"tpr|{name}"] = lambda w, c, p, rows=rows: _share(design, w, rows, True)
        stats[f"fpr|{name}"] = lambda w, c, p, rows=rows: _share(design, w, rows, False)
        if name != "none":
            for metric, matches in (("tpr", True), ("fpr", False)):
                stats[f"diff_{metric}|{name}"] = lambda w, c, p, rows=rows, m=matches: (
                    _share(design, w, rows, m) - _share(design, w, arm == "none", m)
                )
    groups = list(dict.fromkeys(calls["group"][design.cluster >= 0]))
    for group in groups:
        if group != reference:
            stats[f"vs_uk|{group}"] = lambda w, c, p, g=group: (
                _share(design, w, (calls["group"] == g).to_numpy(), True)
                - _share(design, w, (calls["group"] == reference).to_numpy(), True)
            )
    pairs = _pairs(design)
    if len(pairs[0]):
        stats["primary"] = lambda w, c, p: _paired_mean(design, w, c, p, pairs)
    return stats


def _pairs(design: _Design) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of the woman's-name and the man's-name arm for each matching profile and
    group that has both with valid answers."""
    calls = design.calls
    named = calls[(calls["kind"] == "name") & design.match & design.valid]
    key = ["pid", "group"]
    women = named[named["name_gender"] == "female"].reset_index().set_index(key)["index"]
    men = named[named["name_gender"] == "male"].reset_index().set_index(key)["index"]
    both = women.index.intersection(men.index)
    return women.loc[both].to_numpy(), men.loc[both].to_numpy()


def _paired_mean(design, weights, clusters, profiles, pairs) -> float:
    women, men = pairs
    weight = (
        profiles[design.profile[women]]
        * clusters[design.cluster[women]]
        * clusters[design.cluster[men]]
    )
    total = weight.sum()
    if not total:
        return np.nan
    return float((weight * (design.yes[women] - design.yes[men])).sum() / total)


def contrasts(
    calls: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED, reference: str = REFERENCE_GROUP
) -> dict:
    """Every rate and difference with a 95% percentile interval and a bootstrap p.

    The bootstrap resamples profiles and, within each origin group and gender, first names. Keys
    of `statistics`: `tpr|<arm>`, `fpr|<arm>`, `diff_tpr|<arm>` and `diff_fpr|<arm>` (an arm minus
    `none`), `vs_uk|<group>` (the TPR of a group's name arms minus the reference group's) and
    `primary` (women's names minus men's names, TPR, paired by profile and group). `p` is twice
    the smaller share of resamples on either side of zero, counting the observed value as one
    more resample. `holm` adjusts `vs_uk`, `diff_tpr|swap` and `diff_tpr|neutral` together.
    """
    design = _Design(calls)
    stats = _statistics(design, reference)
    rng = np.random.default_rng(seed)
    point = {name: f(*design.weights(None)) for name, f in stats.items()}
    runs = {name: np.empty(n_boot) for name in stats}
    for i in range(n_boot):
        draw = design.weights(rng)
        for name, f in stats.items():
            runs[name][i] = f(*draw)
    report = {}
    for name, value in point.items():
        sample = runs[name][~np.isnan(runs[name])]
        low, high = np.percentile(sample, [2.5, 97.5]) if len(sample) else (np.nan, np.nan)
        below, above = (1 + (sample <= 0).sum()), (1 + (sample >= 0).sum())
        report[name] = {
            "value": value,
            "ci_low": float(low),
            "ci_high": float(high),
            "p": float(min(1.0, 2 * min(below, above) / (len(sample) + 1))),
        }
    family = [
        n for n in report if n.startswith("vs_uk|") or n in ("diff_tpr|swap", "diff_tpr|neutral")
    ]
    adjusted = holm([report[n]["p"] for n in family])
    return {"statistics": report, "holm": dict(zip(family, adjusted, strict=True))}


def name_ranges(
    calls: pd.DataFrame, n_shuffle: int = 200, seed: int = SEED, min_cells: int = 10
) -> dict:
    """For each name arm cell (`<group>|<gender>`), the range of the TPR across first names with
    at least `min_cells` valid answers on matches, and the range that shuffling the first names
    across the cell's answers gives (`shuffled_mean` and `shuffled_p95` over `n_shuffle`)."""
    rng = np.random.default_rng(seed)
    named = calls[(calls["kind"] == "name") & calls["match"] & calls["yes"].notna()]
    ranges = {}
    for (group, gender), rows in named.groupby(["group", "name_gender"], sort=False):
        size = rows.groupby("first")["yes"].transform("size")
        rows = rows[size >= min_cells]
        if rows["first"].nunique() < 2:
            continue
        names, yes = pd.factorize(rows["first"])[0], rows["yes"].to_numpy()

        def spread(labels, yes=yes):
            means = np.bincount(labels, yes) / np.bincount(labels)
            return float(means.max() - means.min())

        shuffled = [spread(rng.permutation(names)) for _ in range(n_shuffle)]
        ranges[f"{group}|{gender}"] = {
            "range": spread(names),
            "n_names": int(rows["first"].nunique()),
            "shuffled_mean": float(np.mean(shuffled)),
            "shuffled_p95": float(np.percentile(shuffled, 95)),
        }
    return ranges
