# hiring-bias-xai

**Does a CV-screening model treat women and men differently, and which words make it do so?**

An explainable-AI audit of a text classifier that screens candidates by occupation, on 99,069 real biographies.

<p align="center">
  <img src="results/figures/f1_gap_vs_share.png" width="520" alt="Scatter plot of 28 occupations: the main model's TPR gap rises with the share of women in the occupation, r = 0.85.">
</p>

## Key findings so far

- **The model mirrors the imbalance in its training data.** In occupations where most people are men, it misses more women; where most are women, it misses more men. The gap tracks the share of women with r = 0.85 [0.71, 0.93], and 10 of 28 occupations have a gap that survives a Holm correction.
- **Pronouns carry part of that gap.** Among surgeon bios, the words *she*, *her*, *he* and *his* alone put women's scores 1.12 log-odds [1.02, 1.24] below men's.
- **Not every word behind the gap is a gender word.** In the "model" occupation the largest single contribution comes from *freeones*, the name of a listing site for adult performers. It appears in 54 of 150 women's bios and in none of the 150 men's.
- **Pronouns move scores more than decisions.** Turning *she* into *he* (and back) changes 6.9% of the main model's decisions, about as many as simply deleting the pronouns (6.3%). The baseline model reacts more: 13.5% against 8.8%.
- **The explanations are reliable in aggregate, not word by word.** Removing the words Shapley ranks highest hurts the main model far more than removing random words, so the values point at what matters. But the exact order of the words in one bio changes with how the values are computed. The conclusions above rest on totals over hundreds of bios.
- **A better model is not a fairer one.** The transformer beats the TF-IDF baseline on macro-F1 (0.783 vs 0.728), but its bias metric is not lower beyond noise.

## Why it matters

A job platform that ranks candidates with a model is a high-risk AI system under the EU AI Act (Annex III). The platform has to show that the ranking does not depend on gender, and if it does, why. Accuracy alone cannot answer that; explanations can point at the words responsible and test whether removing them helps.

## What the audit does

| Question | Method | Kind | Status |
|---|---|---|---|
| Who is treated differently, and how much? | TPR gap per occupation, bootstrap CIs, permutation test | Bias metric | Done |
| Which words drive one decision? | Word-level Shapley values (permutation sampling) | Local | Done |
| Which words make up the gap between women and men? | Exact decomposition of the score gap into word contributions | Global | Done |
| Does the decision change if only the gender changes? | Counterfactual pronoun swaps, with a deletion control | Local and global | Done |
| Can we trust these explanations? | Deletion curves, seed and method stability | Explanation check | Done |
| Where does the model store gender? | Probing and activation patching | Mechanistic | Done |
| Does removing the cues close the gap, and at what cost? | Retraining on scrubbed text, masking proxy words | Mitigation | Planned |
| Does a general-purpose LLM do the same with names? | Black-box audit with names from four origin groups | Audit | Planned |

Full results, every figure and the reasoning behind each number are in **[docs/results.md](docs/results.md)**.

## Results

Test split (99,069 bios), 95% bootstrap intervals.

| Model | Macro-F1 | Accuracy | RMS TPR gap |
|---|---|---|---|
| TF-IDF + logistic regression | 0.728 [0.724, 0.733] | 0.795 [0.793, 0.798] | 0.167 [0.151, 0.190] |
| DeBERTa-v3-small, fine-tuned | **0.783** [0.779, 0.788] | **0.837** [0.835, 0.839] | 0.148 [0.134, 0.172] |

The RMS TPR gap summarises the gaps of all 28 occupations in one number; 0 would mean equal opportunity. A random reassignment of gender gives about 0.04.

## Data and models

- **Data**: [Bias in Bios](https://huggingface.co/datasets/LabHC/bias_in_bios) (De-Arteaga et al., 2019, MIT licence): English biographies labelled with one of 28 occupations and a binary gender inferred from pronouns. Details in the [data card](docs/data_card.md).
- **Models**: a TF-IDF plus logistic regression baseline, and DeBERTa-v3-small fine-tuned on a 60,000-bio subsample (the main model). Explanations run on a fixed showcase subset of 1,721 test bios from six occupations: surgeon, software engineer and composer (mostly men), dietitian, nurse and model (mostly women).
- **Decisions** are recorded in [docs/decisions/](docs/decisions/); terms are defined in the [glossary](GLOSSARY.md).

## Repository layout

```text
src/hiring_bias_xai/   library: data, models, fairness metrics, attribution, plotting
scripts/               one entry point per experiment; each writes to results/
results/               metrics (JSON, CSV) and figures (PDF, PNG)
docs/                  results, data card, decision records, figure guidelines
tests/                 pytest suite, run by CI on every pull request
```

## How to run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The lock file pins PyTorch to the CUDA 12.8 build.

```bash
uv sync
uv run pytest -m "not gpu and not slow"
uv run python scripts/prepare_data.py      # downloads the data at a pinned revision
uv run python scripts/train_baseline.py    # CPU, about 5 minutes
uv run python scripts/train_deberta.py     # GPU, about 3.5 hours on an RTX 3050 (two learning rates)
```

Every other script in `scripts/` reproduces one result or figure; [docs/results.md](docs/results.md) names the script next to each. Data and model weights live outside the repository, in `HBX_ARTIFACTS` (default `~/.cache/hiring-bias-xai`). Without `uv`: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128`, then `pip install -e . --no-deps`.

## License

[MIT](LICENSE). The dataset keeps its own licence.
