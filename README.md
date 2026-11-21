# hiring-bias-xai

**Does an AI that screens CVs treat women and men differently, and why?**

A hospital needs a surgeon and receives thousands of applications. An AI model reads each candidate's short professional bio first and decides which of 28 jobs it describes (surgeon, nurse, lawyer and so on). Bios it labels "surgeon" go on to the recruiter; the rest are set aside and never seen by a person. A fair screener passes on the same share of real surgeons whether they are women or men.

I built such a model from real professional bios (about 400,000 of them) and checked who it passes on. Among people who really are surgeons, it passes on 76 of every 100 men but only 56 of every 100 women. Nurses show the opposite: 84 of every 100 women, 73 of every 100 men. Looking at all 28 jobs together, the more a job is held by one gender, the more the model favours that gender.

<p align="center">
  <img src="results/figures/f1_gap_vs_share.png" width="520" alt="Each dot is a job. The more women hold a job, the more the model favours women in it; surgeon: 56 of 100 qualified women passed on against 76 of 100 men; nurse: 84 of 100 women against 73 of 100 men.">
</p>

This project uses explainable-AI methods to find out why: which words drive these decisions, whether those explanations can be trusted, and what reduces the gap.

## Key findings so far

- **Gender words push the decisions.** In surgeon bios, *she* and *her* lower the model's score for "surgeon" and *he* and *his* raise it, so together they put women below men. In nurse bios they do the opposite ([details](docs/results.md#f3-which-words-make-up-the-gap-between-women-and-men)).
- **Some words stand in for gender.** For the job "model", the single strongest word is *freeones*, the name of a website listing adult performers. It appears in 54 of 150 women's bios and in none of the 150 men's. The model learned it as a sign of the job, and it only ever shows up for women.
- **Swapping *she* for *he* rarely flips a decision.** It changes the decision for 7 of every 100 bios, about as often as simply deleting the pronouns does (6 of 100). The simpler baseline model reacts more: 14 against 9 of every 100.
- **The explanations hold in aggregate, not word by word.** Removing the words the explanation ranks highest hurts the model far more than removing random words, so the explanations point at what matters. But the exact order of the words in one bio changes with how the explanation is computed, so the conclusions rest on totals over hundreds of bios.
- **Removing the pronouns narrows the gap by a fifth and leaves most of it.** Training and scoring on bios without pronouns brings the gap between women and men from 0.143 to 0.115 for half a point of macro-F1, still more than twice what chance gives. I would recommend it as a first step, not as a fix: in the job "model" the model passes on 40 more qualified women than men per 100 of each, against 44 before. Deleting the six words that stand in for gender barely moves it ([details](docs/results.md#f7-does-removing-the-cues-close-the-gap)).
- **A local LLM screener does not react to the gender or origin of a name, but it reacts to there being a name.** With a woman's name instead of a man's it says yes to the same share of qualified candidates, 53 of every 100 either way, and names from Spain, Morocco, Romania, Hispanic America and the United Kingdom give the same answers. Any name at all lowers the yes-rate by about 6 points against no name, and rewording the instruction moves it by 12 to 20 points. The intervals rule out gaps of 5 points or more, not smaller ones ([details](docs/results.md#f8-does-an-llm-screener-react-to-names)).
- **A more accurate model is not a fairer one.** The transformer is right more often than a simple word-count baseline (84 against 80 of every 100 bios), but its gap between women and men is not clearly smaller.

## Why it matters

A job platform that ranks candidates with a model is a high-risk AI system under the EU AI Act (Annex III). The platform has to show that the ranking does not depend on gender, and if it does, why. Accuracy alone cannot answer that; explanations can point at the words responsible and test whether removing them helps.

## What the audit does

| Question | Method | Kind | Status |
|---|---|---|---|
| Who is passed on less often, and by how much? | Gap in the share of qualified women and men passed on (TPR gap), bootstrap intervals, permutation test | Bias metric | Done |
| Which words drive one decision? | Word-level Shapley values (permutation sampling) | Local | Done |
| Which words make up the gap between women and men? | Exact decomposition of the score gap into word contributions | Global | Done |
| Does the decision change if only the gender changes? | Counterfactual pronoun swaps, with a deletion control | Local and global | Done |
| Can the explanations be trusted? | Deletion curves, seed and method stability | Explanation check | Done |
| Where does the model store gender? | Probing and activation patching | Mechanistic | Done |
| Does removing the cues close the gap, and at what cost? | Retraining on scrubbed text, masking proxy words | Mitigation | Done |
| Does a general-purpose LLM do the same with names? | Black-box audit with names from five origin groups | Audit | Done |

Full results, every figure and the reasoning behind each number are in **[docs/results.md](docs/results.md)**.

## Results

Tested on 99,069 bios the models never saw, with 95% bootstrap intervals.

| Model | Right job, out of 100 bios | Macro-F1 | Gender gap (RMS TPR gap) |
|---|---|---|---|
| Word counts + logistic regression (baseline) | 79.5 [79.3, 79.8] | 0.728 [0.724, 0.733] | 0.167 [0.151, 0.190] |
| DeBERTa-v3-small, fine-tuned (main model) | **83.7** [83.5, 83.9] | **0.783** [0.779, 0.788] | 0.148 [0.134, 0.172] |

Macro-F1 averages the quality over the 28 jobs, so small jobs count as much as large ones. The gender gap sums up, over all 28 jobs, how far the share of qualified women passed on is from the share of qualified men: 0 would be equal treatment, and shuffling the genders at random gives about 0.04.

## Data and models

- **Data**: [Bias in Bios](https://huggingface.co/datasets/LabHC/bias_in_bios) (De-Arteaga et al., 2019, MIT licence): English biographies labelled with one of 28 occupations and a binary gender inferred from pronouns. Details in the [data card](docs/data_card.md).
- **Models**: a TF-IDF plus logistic regression baseline, and DeBERTa-v3-small fine-tuned on a 60,000-bio subsample (the main model). Explanations run on a fixed showcase subset of 1,721 test bios from six occupations: surgeon, software engineer and composer (mostly men), dietitian, nurse and model (mostly women).
- **Decisions** are recorded in [docs/decisions/](docs/decisions/); terms are defined in the [glossary](GLOSSARY.md).

## Repository layout

```text
src/hiring_bias_xai/   library: data, models, fairness metrics, attribution, plotting
scripts/               one entry point per experiment; each writes to results/
results/               metrics (JSON, CSV) and figures (PDF, PNG)
notebooks/audit.ipynb  the whole audit told from the saved results; CI runs it
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
