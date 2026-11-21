# 0003. Attribution: word-level Shapley values

Status: accepted. The pilots and the run are done; results are at the end.

## Context

The audit asks which words pushed a screener towards an occupation. The answer feeds the gender-gap decomposition ([ADR 0004](0004-gap-decomposition.md)), which needs the values of a bio to add up to its score. The explanations run on the showcase subset: 1,721 test bios.

Text has no neutral bio to compare against, and the main model, DeBERTa-v3, was pretrained without `[MASK]`. An explanation method has to say what a missing word is, and the answer changes the model's behaviour.

## Decision

Word-level Shapley values by permutation sampling (Lundberg and Lee 2017; Mitchell et al. 2022).

- **Score**: the log-odds of the bio's true occupation against all others, `z_c - logsumexp(z_k, k != c)`, computed from the logits. Probabilities saturate to exactly 0 and 1 in float32; log-odds do not. Both screeners provide logits (the baseline model through `decision_function`, in float64).
- **Units**: whitespace-separated words. Removal acts on the text, so all sub-words of a word go together.
- **Removal**: `delete` drops the word and rejoins with single spaces. `unk` replaces each removed word with one `[UNK]`. The main model reads `[UNK]` as its unknown token (checked: `alpha [UNK] beta` becomes `▁alpha`, `[UNK]`, `▁beta`). For the baseline model's TF-IDF it becomes the word "unk", which is not in the vocabulary of the fitted model (97,315 entries, no entry with "unk"). So for the baseline model `unk` ignores the word but breaks the bigram that spans it, while `delete` creates a new bigram across the gap.
- **Sampling**: each random order of the words comes with its reverse (antithetic pairs). An order walks from the empty bio to the full bio, adding one word at a time, so the values of every order add up to `score(full) - score(empty)`, and so does their average. `n_permutations` counts the random orders; each is paired, so a bio gets twice that many walks.
- **Cost control**: all coalitions of a bio go to the classifier in one call with identical texts merged, and the main model sorts a call's texts by token length before batching.
- **Storage**: per bio, the sampled orders and the 28 logits of every evaluated coalition (float32), in an npz file under `artifacts_dir()/attributions/<screener>/`. The values for any other occupation, such as a rival occupation in the decomposition, are recomputed from those logits without running the model again.
- **Removal choice and permutation count are measured, not assumed.** A removal pilot and a cost pilot (below) set them, and `results/shapley_pilot.json` records both with their rules.

## Alternatives considered

- **Integrated Gradients** (Sundararajan et al. 2017). It needs a reference input, and no bio is neutral. `[PAD]` and `[MASK]` references are out of distribution for DeBERTa-v3, so the path from reference to bio crosses inputs the model never saw. The completeness axiom also holds only for that arbitrary reference.
- **Leave-one-out occlusion.** Cheap (one evaluation per word), but it measures each word with all the others present. Two words that carry the same information each get a small value, because the other one still covers for it. It stays as the cheap comparison in the explanation checks, and as the fallback if the projected Shapley run is too slow.
- **LIME.** A local linear surrogate fitted to random perturbations. Its output depends on the sample and the kernel, so two seeds can rank words differently, and the fidelity of the surrogate would need its own check.
- **Attention weights.** Attention shows where the model looked, not what moved the decision (Jain and Wallace 2019; Wiegreffe and Pinter 2019). Heads and layers also give no single number per word.
- **A `[MASK]` removal.** Not offered, for the reason against a `[MASK]` reference.

## Consequences

- Cost is the bottleneck. Each bio needs about `2 x n_permutations x (words + 1)` coalition evaluations before merging identical texts.
- The values are exact only in the limit of many orders. The cost pilot measures how far two seeds agree.
- Values depend on the removal. The secondary removal is run in the explanation checks to show how much.
- The log-odds target is the true occupation, so a wrong screening decision is explained by what pulled the score of the right occupation down.
- Files are local artifacts read back to resume a run. They are written through a temporary file and renamed, and loaded with pickle disabled; a damaged or stale file is scored again.

## Pilot design

- **Removal pilot.** 200 random dev bios; for each, 30% of the words are dropped at random with each removal (same words for both), 5 draws, main model. The removal with the smaller mean macro-F1 drop is primary. A tie goes to `delete`.
- **Cost pilot.** 50 random showcase-subset bios, primary removal, 5, 10, 20 and 40 permutations, seeds 0 and 1 (40 was added after 20 reached a median of 0.83, below the rule, with the projected run at 0.8 of the 3 hours). Seed agreement is the median Spearman correlation of a bio's two value vectors. The permutation count is the smallest with median at least 0.9. If the projection (seconds per bio times 1,721) exceeds 3 hours, the run falls back to leave-one-out occlusion and this record says why.

## Inference throughput

Measured on the RTX 3050 (4 GB) with 500 dev bios, the main model in inference mode with bfloat16 autocast, after a warm-up. Texts are sorted by token length.

| Batch | Sequences per second | Peak allocated | Peak reserved |
|---|---|---|---|
| 8 | 411 | 637 MB | 684 MB |
| 16 | 598 | 718 MB | 868 MB |
| 32 | 655 | 844 MB | 1,100 MB |
| 64 | 644 | 1,190 MB | 1,742 MB |

The rule was the largest batch with peak allocated memory under 2,500 MB, so the run uses 64. Throughput is flat from 32 up. Training peaked at 2,863 MB allocated and the driver failed at 3,120 MB (ADR 0002); both are far above this.

A smoke run on 4 showcase bios (40 to 103 words) with 5 permutations took 0.66 s per bio with `delete` and 1.06 s with `unk`; the efficiency error was 2e-15.

## Pilot results

Files: `results/shapley_pilot.json` (both pilots, with their rules), `results/shapley_stability.json`, `results/shapley_summary.json`, `results/f2_cases.json`.

**Removal pilot.** Main model, 200 dev bios, unperturbed macro-F1 0.705. Dropping 30% of the words, 5 draws:

| Removal | Mean macro-F1 drop | Draws |
|---|---|---|
| `delete` | 0.088 | 0.053, 0.114, 0.078, 0.132, 0.065 |
| `unk` | 0.132 | 0.094, 0.164, 0.173, 0.179, 0.051 |

`delete` disturbs the model less, so it is the primary removal and `unk` is the comparison in the explanation checks. The draws overlap: `unk` was worse in four of the five draws, and the gap is 0.04 on 200 bios, so the choice is supported, not settled.

**Cost pilot.** Main model, 50 showcase-subset bios, `delete`, seeds 0 and 1:

| Permutations | Median Spearman between seeds | Seconds per bio | Evaluations per bio |
|---|---|---|---|
| 5 | 0.607 | 0.45 | 534 |
| 10 | 0.717 | 0.88 | 1,061 |
| 20 | 0.834 | 1.73 | 2,108 |
| 40 | 0.889 | 3.42 | 4,178 |

**The 0.9 rule was not met.** No count reached a median of 0.9; 40 came closest at 0.889. The pilot chose 40 under its fallback (the largest count tried), and the rule's other branch did not apply: the projection for 1,721 bios was 1.64 h, under the 3 h limit, so no fallback to leave-one-out occlusion. The first pilot tried 5, 10 and 20 only; 20 reached 0.834 with a projection of 0.83 h, so 40 was added before looking at more data. The full run took 7,076 s (1.97 h) for the main model, 20% above the projection, and 272 s for the baseline model, 4,489 evaluations per bio, 1,721 bios each, mean 58.2 words. The efficiency error is at most 9e-15 and 1.4e-14.

**What the shortfall means.** The Spearman correlation counts all words of a bio, and most have values near zero. Two seeds order those words differently while agreeing on the words that matter. The check on the baseline model (`scripts/shapley_stability.py`, same 50 bios and seeds, `delete`) supports this only partly:

| Permutations | All words | Top 10 words by size | Other words | Overlap of the 8 largest |
|---|---|---|---|---|
| 5 | 0.821 | 0.879 | 0.679 | 0.72 |
| 10 | 0.893 | 0.915 | 0.797 | 0.78 |
| 20 | 0.925 | 0.927 | 0.863 | 0.83 |
| 40 | 0.962 | 0.958 | 0.931 | 0.88 |

The other words agree less than the top ten at every count, but at 40 permutations the gap is 0.03, and on average 7 of the 8 largest words are the same in both seeds. Two caveats. The baseline model is more stable than the main model (0.962 against 0.889 for all words at 40), so these numbers are an upper bound for the main model, whose two-seed values for the run are not stored and were not rechecked. And the top-10 correlation has only ten words, so it is noisy per bio.

So for the main model, per-word values, and especially the ranking of low-value words, carry sampling noise that a single bio does not average out. The word-level claims that follow need many bios: the decomposition aggregates over the 1,721 bios and over word groups such as gender words, where the noise averages out, and the explanation checks report stability. F2 shows single bios and says so in its caption.

**Throughput.** The table above is the measurement behind batch 64. The pilot's own peak allocated memory on the main model was 1,142 MB. The run had no driver errors.

## Limit of whitespace units

A unit is a whitespace-separated word, so punctuation can glue two words into one unit: `shoot...she`, `Humor/Romance`, `she/her/hers`. `scripts/word_units.py` counts them on the 1,721 showcase bios (`results/word_units.json`). Of 100,246 units, 1,171 (1.17%) are glued: letters on both sides of punctuation inside the unit, apostrophes excepted. Of those, 14 contain a gender word. Counting the 4,986 units that are a gender word on their own, 14 of 5,000 gendered units (0.28%) are glued. A finer unit would split these 14 and leave the other 4,986 alone, so it would change at most 0.28% of the gendered units, under the 1% at which a rerun would be worth its two hours of GPU. Glued units with no gender word are split in a finer unit too, and their values are shared between two words. The run was not repeated with a finer unit.

A gender word is a unit whose core (lower case, punctuation at its ends removed) is one of the words `data.scrub` removes. `she,` and `(her)` count; `sheriff` and `shoot...she` do not.

## F2: how the two bios are picked

F2 shows two showcase-subset bios, main model, with the eight units of largest absolute Shapley value. Values are recomputed for the predicted occupation from the stored logits, so the run's target (the true occupation) does not limit the choice. The cases mirror the finding of F1, that errors follow the gender imbalance of the occupation. The rules were fixed before looking at the bios:

- **Woman in a male-dominated occupation**: among women whose true occupation is one of the three male-dominated showcase occupations and whom the main model gets wrong, the bio where the gendered units add the most to the log-odds of the predicted occupation (largest sum of their Shapley values, above zero).
- **Man in a female-dominated occupation**: the same for men in the three female-dominated showcase occupations.
- Ties go to the bio that comes first in the subset. Bios under 15 word units are never picked, nor are bios with a run of six or more digits, which may hold a licence or registry number. A bio is skipped when the stored full-bio logits predict another occupation than the saved prediction (1 of 1,721). If a group has no qualifying bio, the figure and `results/f2_cases.json` say so and no other rule replaces it.

A first version of F2 used the largest sum over all misclassified bios and the largest gender-word share among correct ones. Review showed that it picked outliers, a fan-fiction listing and an advertisement, and that its gendered-word match found `she` inside `shoot...she`. The rules above replace it.

Picks (`results/f2_cases.json`): among 114 misclassified women in the male-dominated occupations, `test-30717`, true composer, predicted rapper, the 11 gendered units add +3.21 (131 units). Among 154 misclassified men in the female-dominated ones, `test-9467`, true model, predicted poet, the 2 gendered units add +1.98 (51 units). In both bios the largest single words are topic words (songs, career, play), not gendered ones: the gendered units add up because there are several of them, and one gendered unit alone is +0.6 and +1.0. The picks are the largest of their groups, not typical. F2 shows no interval per word, because the run stored one seed.

## Update from the explanation checks: checks not run

Weight randomization (Adebayo et al. 2018) was first skipped on the argument that a perturbation method reads only the model's outputs, so it passes by construction. That argument was wrong: Shapley reads the outputs of whatever model it is given, and a random network gives different outputs. The model randomisation run checks it; the section below holds the expectation. Retraining-based evaluation (ROAR; Hooker et al. 2019) is not run: it needs one fine-tuning run for each fraction of removed words, and the two learning-rate runs of the main model took about 3.5 hours (README, "How to run").

## Model randomisation

The check: re-draw every weight of the main model with the initialiser of its own Hugging Face class (seed 42), score the same 100 stability bios with the main run's settings (`delete`, 40 random orders, the orders of the main run), and compare the word ranking with the trained model's. The trained weights are not touched. The run is `deberta-random` in `scripts/stability_runs.py`; the comparison is `randomisation` in `results/stability.json`: median Spearman correlation and median top-10 overlap, with 95% bootstrap intervals over bios. I report the numbers and what they mean, with no threshold.

**Expected outcome, written before the run.** Both medians near 0, against 0.89 and 0.80 for seed 7 against seed 42. A random network has learned nothing about occupations, so its scores do not depend on what the trained model learned, and a ranking of words that followed the trained one would show that the explanation reflects something other than the model. The overlap may sit a little above 0, because a bio's own structure (repeated words, long words) can give some words larger values under any model.

**Limits.** I randomise all weights at once, not layer by layer, so a result says whether the explanation depends on the trained model at all, not where. Randomising the labels, the second version of the check, needs a new fine-tuning run of about two hours (the scrubbed model took 1.9 hours). I do not run it.

## Outcomes

The run finished for both screeners on all 1,721 showcase bios, with `delete` as the removal and 40 random orders per bio (each also read backwards). The main model took 7,076 s and the baseline model 272 s. The agreement rule of 0.9 between two seeds was not met (0.889 for the main model), and the record above says what that costs: values of single words are indications, and the findings rest on totals over many bios. The values fed the gap decomposition (ADR 0004), F2 and the checks of F5. Those checks found that the words ranked highest drive the main model's decisions, and that the exact order of the words depends on the seed (rank correlation 0.89), on how a word is removed (0.47) and on the method (0.42 against leave-one-out). With every weight redrawn (seed 42), the word ranking no longer follows the trained model's: median Spearman correlation 0.062 [0.021, 0.094] and median top-10 overlap 0.2 [0.2, 0.2] (mean 0.22 [0.188, 0.252]), on the 100 stability bios, against 0.89 and 0.80 for seed 7. The correlation is near 0 as expected. The overlap is 0.2, not 0, and I should have written chance level, not 0: a random ranking shares on average 10 divided by the bio's word count of its top words, which is 0.198 over these bios (computed from the saved word counts, not stored in the json). So the overlap is what chance gives. The explanations follow what the model learned. Retraining-based evaluation was not run.
