# hiring-bias-xai

**Does an AI that screens CVs treat women and men differently, and why?**

A hospital needs a surgeon and receives thousands of applications. An AI model reads each candidate's short professional bio first and decides which of 28 jobs it describes (surgeon, nurse, lawyer and so on). Bios it labels "surgeon" go on to the recruiter; the rest are set aside and never seen by a person. A fair screener passes on the same share of real surgeons whether they are women or men.

I trained such a model on 60,000 real professional bios and checked who it passes on among 99,069 bios it had never seen. Among people who really are surgeons, it passes on 76 of every 100 men but only 56 of every 100 women. Nurses show the opposite: 84 of every 100 women, 73 of every 100 men. Looking at all 28 jobs together, the more a job is held by one gender, the more the model favours that gender.

<p align="center">
  <img src="results/figures/f1_gap_vs_share.png" width="520" alt="Each dot is a job. The more women hold a job, the more the model favours women in it; surgeon: 56 of 100 qualified women passed on against 76 of 100 men; nurse: 84 of 100 women against 73 of 100 men.">
</p>

This project uses explainable-AI methods to find out why: which words drive these decisions, whether those explanations can be trusted, and what reduces the gap.

## Key findings

- **Gender words and stand-in words push the decisions ([F3](docs/results.md#f3-which-words-make-up-the-gap-between-women-and-men)).** In surgeon bios, *she* and *her* lower the model's score for "surgeon" and *he* and *his* raise it, which puts women below men; nurse bios show the reverse. For the job "model" the strongest single word is *freeones*, the name of a website listing adult performers. It is in 54 of 150 women's bios and in none of the 150 men's, so the model learned it as a sign of the job that only women's bios show.
- **Swapping *she* for *he* changes few decisions ([F4](docs/results.md#f4-does-changing-only-the-gender-change-the-decision)).** It changes the decision for 7 of every 100 bios, about as often as deleting the pronouns does (6 of 100). The simpler baseline model reacts more: 14 against 9 of every 100.
- **The explanations hold in aggregate and depend on the method word by word ([F5](docs/results.md#f5-can-the-explanations-be-trusted)).** Removing the words an explanation ranks highest hurts the model far more than removing random words, so the explanations point at what matters. The exact order of the words in one bio changes with how the explanation is computed, so the conclusions rest on totals over hundreds of bios.
- **Gender stays inside the model after the pronouns go ([F6](docs/results.md#f6-gender-inside-the-model)).** A simple probe guesses gender from the main model's internal states for 58% to 63% of bios, against 52% from the job alone. The effect of swapping pronouns moves from the swapped words to the rest of the bio within a few layers.
- **Removing the pronouns narrows the gap by a fifth and leaves most of it ([F7](docs/results.md#f7-does-removing-the-cues-close-the-gap)).** Training and scoring on bios without pronouns brings the gender gap from 0.143 to 0.115 for half a point of macro-F1, and what remains is still more than twice what chance gives. I would recommend it as a first step, not as a fix: in the job "model" the model still passes on 40 more qualified women than men per 100 of each, against 44 before. Deleting the six words that stand in for gender barely moves it.
- **A local LLM screener answers alike for women's and men's names and for every origin, but any name changes its answers ([F8](docs/results.md#f8-does-an-llm-screener-react-to-names)).** With a woman's name instead of a man's, its yes-rate for qualified candidates differs by -0.5 points (interval -4.7 to +3.5). Names from Spain, Morocco, Romania, Hispanic America and the United Kingdom give the same answers. Any name lowers the yes-rate by about 6 points against no name, and rewording the instruction changes 13% and 16% of its decisions.
- **A more accurate model is not a fairer one.** The transformer is right more often than a simple word-count baseline (84 against 80 of every 100 bios), but its gender gap is not clearly smaller.

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

Every result, figure and the reasoning behind each number are in **[docs/results.md](docs/results.md)**. Each technique has an entry in **[docs/methods.md](docs/methods.md)**, and the models and their limits are in the **[model card](docs/model_card.md)**.

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
docs/                  results, model card, methods, data card, decision records
docs/model_card.md     the three screeners and the LLM: recipe, metrics, limits
docs/methods.md        one entry per technique: what it does, how to read it, pitfalls
tests/                 pytest suite, run by CI on every pull request
```

## How to run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The lock file pins PyTorch to the CUDA 12.8 build. Run the steps in this order; times are measured on the RTX 3050 Laptop GPU (4 GB). Data and model weights live outside the repository, in `HBX_ARTIFACTS` (default `~/.cache/hiring-bias-xai`). Without `uv`: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128`, then `pip install -e . --no-deps`.

| Step | Commands, in order | Time |
|---|---|---|
| Set up | `uv sync`; `uv run pytest -m "not gpu and not slow"` | install not recorded; tests about 3 minutes |
| Data | `scripts/prepare_data.py` (pinned revision) | not recorded |
| Models | `scripts/train_baseline.py`; `scripts/train_deberta.py` | 5 minutes on the CPU; 3.45 hours on the GPU for two learning rates |
| F1 | `scripts/gap_statistics.py`; `scripts/figure_f1.py` | not recorded |
| F2, F3 | `scripts/shapley_pilot.py`; `scripts/shapley_run.py`; `scripts/shapley_stability.py`; `scripts/word_units.py`; `scripts/figure_f2.py`; `scripts/gap_decomposition.py`; `scripts/figure_f3.py` | run 1.97 hours for the main model and 5 minutes for the baseline; decomposition about 30 seconds; the rest not recorded |
| F4 | `scripts/fetch_first_names.py`; `scripts/counterfactual_hand_check.py`; `scripts/counterfactual_gender.py`; `scripts/figure_f4.py` | scoring about 45 seconds; the rest not recorded |
| F5 | `scripts/deletion_curves.py`; `scripts/stability_runs.py`; `scripts/stability.py`; `scripts/figure_f5.py` | curves about 6 minutes; the two reruns about 10 and 15 minutes; the rest not recorded |
| F6 | `scripts/probing_features.py`; `scripts/probing.py`; `scripts/patching.py`; `scripts/figure_f6.py` | features about 1 minute each; probes 1, 4 and 5 minutes; patching 32 seconds |
| F7 | `scripts/proxy_words.py`; `scripts/train_scrubbed.py`; `scripts/mitigation.py`; `scripts/figure_f7.py` | training about 1.9 hours; scoring about 2.8 minutes per condition and 2 minutes for the report; the rest not recorded |
| F8 | `scripts/llm_names.py`; `scripts/llm_name_check.py`; `scripts/llm_pilot.py`; `scripts/llm_run.py`; `scripts/llm_analyse.py`; `scripts/figure_f8.py` | name check 3 minutes; pilot 2.4 minutes; run 2.99 hours; analysis about 25 seconds; the rest not recorded |

Each figure is drawn from small files in `results/`. This table names the script that computes them and the one that draws the figure.

| Figure | Result | Computed by | Writes | Drawn by |
|---|---|---|---|---|
| F1 | TPR gap by job | [`gap_statistics`](scripts/gap_statistics.py) | [`gap_statistics.json`](results/gap_statistics.json) | [`figure_f1`](scripts/figure_f1.py) |
| F2 | Shapley values of one decision | [`shapley_run`](scripts/shapley_run.py) | [`shapley_summary.json`](results/shapley_summary.json) | [`figure_f2`](scripts/figure_f2.py) |
| F3 | Gap decomposition by word | [`gap_decomposition`](scripts/gap_decomposition.py) | [`gap_decomposition.csv`](results/gap_decomposition.csv) | [`figure_f3`](scripts/figure_f3.py) |
| F4 | Counterfactual swap | [`counterfactual_gender`](scripts/counterfactual_gender.py) | [`counterfactual_gender.csv`](results/counterfactual_gender.csv) | [`figure_f4`](scripts/figure_f4.py) |
| F5 | Explanation check | [`deletion_curves`](scripts/deletion_curves.py), [`stability`](scripts/stability.py) | [`deletion_curves.csv`](results/deletion_curves.csv), [`stability.json`](results/stability.json) | [`figure_f5`](scripts/figure_f5.py) |
| F6 | Probing and patching | [`probing`](scripts/probing.py), [`patching`](scripts/patching.py) | [`probing.csv`](results/probing.csv), [`patching.csv`](results/patching.csv) | [`figure_f6`](scripts/figure_f6.py) |
| F7 | Mitigation | [`mitigation`](scripts/mitigation.py) | [`mitigation.json`](results/mitigation.json) | [`figure_f7`](scripts/figure_f7.py) |
| F8 | LLM name audit | [`llm_analyse`](scripts/llm_analyse.py) | [`llm_audit.json`](results/llm_audit.json) | [`figure_f8`](scripts/figure_f8.py) |

The arguments of each script and the order inside a step are in [docs/results.md](docs/results.md#reproduce-every-result). Before a pull request, `uv run python scripts/check_docs.py` checks the links and the numbers of this page and the model card against `results/`.

## Cite

- **Data.** De-Arteaga, M. et al. (2019). Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting. FAT*. Version of Ravfogel et al. (2020) on Hugging Face: `LabHC/bias_in_bios`.
- **Model.** He, P., Gao, J. and Chen, W. (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. arXiv:2111.09543.
- **Name lists.** Instituto Nacional de Estadística (INE): Padrón Continuo, 01/01/2022, and Censos de población anuales, 01/01/2025 (CC BY 4.0). Elaboración propia con datos extraídos del sitio web del INE: www.ine.es.
- **LLM audited in F8.** Prism ML, Ternary Bonsai 2 9B (`prism-ml/Ternary-Bonsai-2-9B-gguf`, Apache-2.0), a 2-bit ternary version of Qwen3.5-9B.

## License

[MIT](LICENSE). The dataset keeps its own licence.
