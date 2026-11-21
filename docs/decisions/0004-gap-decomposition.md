# 0004. Gap decomposition: how the score gap splits over words

Status: accepted. Implemented in `src/hiring_bias_xai/gap_decomposition.py`, run by `scripts/gap_decomposition.py`.

## Context

The audit asks which words drive the gap between women's and men's screening scores, not only which words drive one decision. Shapley values are efficient (ADR 0003): the values of a bio add up to its score minus the score of the empty bio, and the empty bio is the same text for everyone. So the difference between the mean scores of women's and men's bios in an occupation splits exactly into one term per word. This describes how the score splits. It does not say what would change if a word changed; that is for the counterfactual work (#8, #10).

## Decision

Per screener and showcase occupation `o`, on the showcase subset, using the stored attributions (target = true occupation `o`):

- **Bios**: those whose true occupation is `o`. Every bio's target is `o`, so all scores are for the same occupation.
- **Word**: `data.word_core` of a unit; a unit with no letter or digit keeps its own text, so no unit is dropped. A bio's value for a word is the sum over its occurrences; a missing word counts 0. A word counts as present in a bio when any unit has that core, even if its value is exactly 0 (the baseline model gives 0 to words outside its vocabulary).
- **Importance** `I(w, o)`: mean value over the bios of `o` that contain `w`, reported only for words in at least 20 bios of `o`.
- **Contribution** `D(w, o)`: mean over women's bios minus mean over men's bios, absent = 0.
- **Check**: the sum of `D` over all words equals the women's mean minus the men's mean of `score(full bio) - score(empty bio)`. The script reports the largest absolute error, which is at float precision (3e-15).
- **Score gap**: the women's mean minus the men's mean of `score(full bio)`, from the stored logits. It differs from the sum of `D` by the difference of the mean empty-bio scores of the two groups. The empty bio is the same text, but the main model's logits for it vary with the batch it was scored in (up to 8e-3 within an occupation, bfloat16), so the drift is not exactly 0: at most 3.7e-4 here, and exactly 0 for the baseline model. Both numbers are reported (`score_gap`, `decomposed_gap`, `empty_score_drift`). Shares use the sum of `D` as the denominator, so the contributions of all words add up to 100%.
- **Top-10 share**: the sum of `D` over the ten words with the largest `|D|`, over the sum of `D`. It can exceed 100% or be negative.
- **Intervals**: 1,000 bootstrap resamples (seed `config.SEED`) of the bios, each gender resampled on its own, with replacement; percentile 2.5 and 97.5. For the top-10 share the ten words are chosen again in each resample (what a reader would see if they reran the study); a second interval keeps the ten words of the full sample (`fixed_ci_*`). The bootstrap resamples bios; it does not resample the Shapley orders, so it does not capture the sampling noise of the values (median seed agreement 0.889 on the main model, ADR 0003). Aggregating over 150 bios per cell averages most of that noise out, but not all.
- **Unstable shares**: a share divides by the total gap, so it is unstable when the total gap is near 0. The rule: the 95% interval of the sum of `D` includes 0. The run prints a warning, `gap_unstable` is true, and F3 gives no share for that occupation. When the total gap is exactly 0 in the sample the share is empty (`null`); a resample with a total of exactly 0 is left out of the percentiles.
- **Usage and weight**: `D = f_F m_F - f_M m_M = (f_F - f_M) * mean(m) + mean(f) * (m_F - m_M)`, with `f` the share of bios that contain the word, `m` its mean value when present, and means over the two genders. The first term is how much more one group uses the word, the second how differently it is weighted. A word absent from a group has no `m` there; its frequency is 0, so the split counts it as 0 and stays exact.
- **Gendered**: `data.is_gendered_word` on the word. The gendered sum is the sum of `D` over those words, with the same bootstrap interval. It needs no division, so it exists when the total gap is near 0. The gendered share is that sum over the sum of `D`.
- **F3**: two panels, the male-dominated and the female-dominated showcase occupation with the largest absolute score gap of the main model. The rule is fixed in `scripts/figure_f3.py` and the picks are recorded in `results/f3_panels.json`. The title states the gendered sum of each panel (below or above men's, or within noise when its interval includes 0), not a share, so it holds where the total gap is not distinguishable from 0.

## Alternatives considered

- **Share over the full-score gap**, as first written in the issue. The two denominators differ by the drift above, which is at most 3.7e-4. The sum of `D` makes the shares add up exactly, so it is the denominator; both gaps are reported.
- **Restricting the top 10 to words in at least 20 bios.** A rare word with a large `|D|` is possible, but a filter would make the top-10 share depend on a second threshold. The table keeps the top 10 words whatever their count and leaves importance empty for them.
- **Bootstrap with the top set fixed only.** It understates the uncertainty, because the choice of the ten words is part of the estimate. Both are reported.
- **Resampling Shapley orders too.** Only one seed was stored per bio. A second run would cost 2 hours of GPU for the main model; not done.

## Consequences

- Shares over 100% are possible and meaningful: the ten largest words add up to more than the gap, and the others pull back.
- With 150 women and 150 men (71 men for dietitian) the intervals of the shares are wide, and unstable ones are not shown.
- The decomposition describes the score, not the model's behaviour under edits.
- Not every large contribution is a property of the person. In the model occupation, the largest word is `freeones`, a listing site. It is in 54 of 150 women's model bios of the showcase subset and in none of the 150 men's (counted on the stored word units), and it adds +0.96 to the main model's +5.40 gap. Part of that gap follows where the bios were scraped from, not gender words. The counts do not show the source of the bios.
- When one group never uses a word (its frequency is 0), the usage and weight parts are each half of the contribution by construction. For such words, such as `he` in the bios of women, read the frequencies and the mean values, not the split.

## Outcomes

The decomposition ran for six showcase occupations and both screeners, and the parts add up to the score gap to within 3e-15. In both screeners the gender words work against women in the male-dominated occupations and for them in the female-dominated ones; the one exception is the main model in software engineer, where the gender words' total cannot be told from zero. For the main model in surgeon the gender words alone put women 1.12 below men; in model they account for 0.80 of a gap of 5.40, and the strongest single word is `freeones`. Shares are not given where the gap itself may be zero (surgeon, software engineer and composer for the main model). The proxy words of ADR 0007 come from this table.
