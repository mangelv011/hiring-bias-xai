# hiring-bias-xai

Explainable-AI audit of a CV-screening text classifier for gender and origin bias.

## Idea

A recruiter searches for an occupation and a model decides which bios match. This project audits that screening decision on Bias in Bios (De-Arteaga et al., 2019), about 396k English biographies over 28 occupations with a binary gender label inferred from pronouns.

Three screeners are compared: a TF-IDF plus logistic regression baseline model, a fine-tuned DeBERTa-v3-small, and a local LLM used as a black-box screener. The audit measures where each one treats women and men differently, then uses explanation methods to ask why.

## Research question

When a screening model favours one gender in an occupation, which parts of the bio drive the difference, and does removing them close the gap?

## Approach

- **Bias metric**: the TPR gap per occupation (TPR women minus TPR men, positive favours women), and its correlation with the share of women in the occupation.
- **Attribution**: word-level Shapley values of the log-odds of the true occupation, by permutation sampling. A word is absent when it is deleted or replaced by `[UNK]`; a pilot picks which.
- **Counterfactual swaps**: pronoun swaps, and name headers built from Spanish INE name lists for four origin groups (Spain, Morocco, Romania, Hispanic America = Colombia, Ecuador and Venezuela).
- **Probing**: gender probes on scrubbed text, with a control task, and activation patching.
- **Sanity checks**: deletion curves and weight randomization.

Expensive explanations run on a fixed showcase subset of the test bios: up to 150 per showcase occupation and gender. Terms are defined in [GLOSSARY.md](GLOSSARY.md); design decisions are in [docs/decisions/](docs/decisions/).

## Status

Package layout, environment, CI, glossary, the first two decision records and the data pipeline (`python scripts/prepare_data.py`, described in [docs/data_card.md](docs/data_card.md)). The fairness metrics with bootstrap CIs and the baseline model (`python scripts/train_baseline.py`, results in `results/metrics_baseline.json`). The main model (`python scripts/train_deberta.py`, results in `results/metrics_deberta.json` and `results/training_log.csv`). The figure style ([docs/figure_guidelines.md](docs/figure_guidelines.md)), the gap statistics (`python scripts/gap_statistics.py`, results in `results/gap_statistics.json`) and figure F1 (`python scripts/figure_f1.py`). The Shapley code is in `src/hiring_bias_xai/attribution.py`, with two scripts: `scripts/shapley_pilot.py` (removal and cost pilots, writes `results/shapley_pilot.json`) and `scripts/shapley_run.py` (both models on the showcase subset, resumable, writes `results/shapley_summary.json`). Their results are not in yet; see [ADR 0003](docs/decisions/0003-shapley-attribution.md).

## Results so far

Test split, 99,069 bios, 95% bootstrap intervals from `src/hiring_bias_xai/fairness.py`.

| Model | Accuracy | Macro-F1 | RMS TPR gap | Gap-share correlation |
|---|---|---|---|---|
| Baseline model | 0.7954 [0.7929, 0.7979] | 0.7282 [0.7235, 0.7332] | 0.1666 [0.1506, 0.1895] | 0.892 [0.823, 0.909] |
| Main model | 0.8370 [0.8346, 0.8391] | 0.7832 [0.7787, 0.7879] | 0.1476 [0.1342, 0.1715] | 0.855 [0.777, 0.873] |

The main model is more accurate: the macro-F1 intervals do not overlap. Its RMS TPR gap is lower, but the intervals overlap, so this data does not show that it is fairer. Details and training history are in [docs/decisions/0002-models.md](docs/decisions/0002-models.md).

### F1: the gap follows the share of women

![The main model's errors follow each occupation's gender imbalance (r = 0.85).](results/figures/f1_gap_vs_share.png)

Each dot is one occupation: its share of women in the training subsample against the main model's TPR gap, with the 95% bootstrap CI of the gap. The six showcase occupations are labelled. A positive gap favours women, so the main model misses more women in male-dominated occupations and more men in female-dominated ones. Under a deuteranopia simulation the highlight and grey dots stay apart ([check](docs/figure_checks/f1_deuteranopia.png)).

### Gap statistics

Computed on the same test split by `scripts/gap_statistics.py`; point values match the tables above to 1e-9. The 28 occupations are the units, so these intervals are wider than the bio-level ones above. The permutation null shuffles gender within each occupation 1,000 times.

| Statistic | Baseline model | Main model |
|---|---|---|
| Pearson r, Fisher-z 95% CI | 0.892 [0.778, 0.949] | 0.855 [0.707, 0.931] |
| Pearson r, occupation bootstrap 95% CI | [0.835, 0.951] | [0.762, 0.933] |
| Spearman rho, occupation bootstrap 95% CI | 0.933 [0.826, 0.969] | 0.830 [0.582, 0.926] |
| Pearson r with one occupation left out | 0.877 (without dietitian) to 0.912 (without model) | 0.838 (without dietitian) to 0.874 (without nurse) |
| RMS TPR gap, observed | 0.1666 | 0.1476 |
| RMS TPR gap, permutation null mean (95th percentile) | 0.0391 (0.0539) | 0.0359 (0.0488) |
| Permutation p-value | 0.001 | 0.001 |
| Occupations with a Holm-adjusted p below 0.05 | 11 of 28 | 10 of 28 |

No shuffle reached the observed RMS TPR gap, so the p-value is the smallest this test can give, 1/1,001. Both models treat women and men unequally in a way chance does not produce, and the correlation with the share of women stays above 0.83 whichever occupation is left out. The main model's lower RMS TPR gap still does not show that it is fairer: the intervals above overlap.

The per-occupation gaps are descriptive. Only the Holm-adjusted ones support a claim that a gap is not zero. For the main model they are: women favoured in dietitian, model, nurse, paralegal, physician and teacher; men favoured in attorney, chiropractor, pastor and surgeon. The other 18 occupations show no gap at that level; that does not show their gaps are zero.

## How to run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The lock file pins PyTorch to the CUDA 12.8 build.

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"   # True on a CUDA machine
uv run pytest
```

Without `uv`, install the same pinned versions, including the CUDA 12.8 build of PyTorch, then the package:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e . --no-deps
```

`requirements.txt` is exported from the lock file with `uv export --no-hashes --no-emit-project -o requirements.txt`.

Data and model weights live outside the repository, in the directory named by the `HBX_ARTIFACTS` environment variable (default `~/.cache/hiring-bias-xai`).

Experiment entry points live in `scripts/` and write small JSON/CSV files and figures to `results/`. `uv run python scripts/prepare_data.py` downloads the dataset at a pinned revision into `HBX_ARTIFACTS` and writes `results/data_summary.json`, `results/showcase_occupations.json` and `results/showcase_subset_ids.csv`; running it twice gives byte-identical files. `uv run pytest -m "not slow"` skips the tests that need the real dataset.

## License

[MIT](LICENSE)
