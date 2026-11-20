# 0003. Attribution: word-level Shapley values

Status: accepted. The pilot results are filled in after the pilots run.

## Context

The audit asks which words pushed a screener towards an occupation. The answer feeds the gender-gap decomposition (#7), which needs the values of a bio to add up to its score. The explanations run on the showcase subset: 1,721 test bios.

Text has no neutral bio to compare against, and the main model, DeBERTa-v3, was pretrained without `[MASK]`. An explanation method has to say what a missing word is, and the answer changes the model's behaviour.

## Decision

Word-level Shapley values by permutation sampling (Lundberg and Lee 2017; Mitchell et al. 2022).

- **Score**: the log-odds of the bio's true occupation against all others, `z_c - logsumexp(z_k, k != c)`, computed from the logits. Probabilities saturate to exactly 0 and 1 in float32; log-odds do not. Both screeners provide logits (the baseline model through `decision_function`, in float64).
- **Units**: whitespace-separated words. Removal acts on the text, so all sub-words of a word go together.
- **Removal**: `delete` drops the word and rejoins with single spaces. `unk` replaces each removed word with one `[UNK]`. The main model reads `[UNK]` as its unknown token (checked: `alpha [UNK] beta` becomes `▁alpha`, `[UNK]`, `▁beta`). For the baseline model's TF-IDF it becomes the word "unk", which is not in the vocabulary of the fitted model (97,315 entries, no entry with "unk"). So for the baseline model `unk` ignores the word but breaks the bigram that spans it, while `delete` creates a new bigram across the gap.
- **Sampling**: each random order of the words comes with its reverse (antithetic pairs). An order walks from the empty bio to the full bio, adding one word at a time, so the values of every order add up to `score(full) - score(empty)`, and so does their average. `n_permutations` counts the random orders; each is paired, so a bio gets twice that many walks.
- **Cost control**: all coalitions of a bio go to the classifier in one call with identical texts merged, and the main model sorts a call's texts by token length before batching.
- **Storage**: per bio, the sampled orders and the 28 logits of every evaluated coalition (float32), in an npz file under `artifacts_dir()/attributions/<screener>/`. The values for any other occupation, such as a rival occupation in #7, are recomputed from those logits without running the model again.
- **Removal choice and permutation count are measured, not assumed.** A removal pilot and a cost pilot (below) set them, and `results/shapley_pilot.json` records both with their rules.

## Alternatives considered

- **Integrated Gradients** (Sundararajan et al. 2017). It needs a reference input, and no bio is neutral. `[PAD]` and `[MASK]` references are out of distribution for DeBERTa-v3, so the path from reference to bio crosses inputs the model never saw. The completeness axiom also holds only for that arbitrary reference.
- **Leave-one-out occlusion.** Cheap (one evaluation per word), but it measures each word with all the others present. Two words that carry the same information each get a small value, because the other one still covers for it. It stays as the cheap comparison in #9, and as the fallback if the projected Shapley run is too slow.
- **LIME.** A local linear surrogate fitted to random perturbations. Its output depends on the sample and the kernel, so two seeds can rank words differently, and the fidelity of the surrogate would need its own check.
- **Attention weights.** Attention shows where the model looked, not what moved the decision (Jain and Wallace 2019; Wiegreffe and Pinter 2019). Heads and layers also give no single number per word.
- **A `[MASK]` removal.** Not offered, for the reason against a `[MASK]` reference.

## Consequences

- Cost is the bottleneck. Each bio needs about `2 x n_permutations x (words + 1)` coalition evaluations before merging identical texts.
- The values are exact only in the limit of many orders. The cost pilot measures how far two seeds agree.
- Values depend on the removal. The secondary removal is run in #9 to show how much.
- The log-odds target is the true occupation, so a wrong screening decision is explained by what pulled the score of the right occupation down.
- Files are local artifacts read back to resume a run. They are written through a temporary file and renamed, and loaded with pickle disabled; a damaged or stale file is scored again.

## Pilot design

- **Removal pilot.** 200 random dev bios; for each, 30% of the words are dropped at random with each removal (same words for both), 5 draws, main model. The removal with the smaller mean macro-F1 drop is primary. A tie goes to `delete`.
- **Cost pilot.** 50 random showcase-subset bios, primary removal, 5, 10 and 20 permutations, seeds 0 and 1. Seed agreement is the median Spearman correlation of a bio's two value vectors. The permutation count is the smallest with median at least 0.9. If the projection (seconds per bio times 1,721) exceeds 3 hours, the run falls back to leave-one-out occlusion and this record says why.

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

To be filled in after the pilots and the run.
