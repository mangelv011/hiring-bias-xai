# hiring-bias-xai

**Does an AI that screens CVs treat women and men differently, and why?** The whole audit is in the [report](report/main.pdf); this page is the short version.

A hospital needs a surgeon and receives thousands of applications. An AI model reads each candidate's short professional bio first and decides which of 28 jobs it describes (surgeon, nurse, lawyer and so on). Bios it labels "surgeon" go on to the recruiter; the rest are set aside and never seen by a person. A fair screener shortlists the same share of real surgeons whether they are women or men.

I trained such a model on 60,000 real professional bios and checked who it shortlists among 99,069 bios it had never seen. Among people who really are surgeons, it shortlists 76 of every 100 men but only 56 of every 100 women. Nurses show the opposite: 84 of every 100 women, 73 of every 100 men. Looking at all 28 jobs together, the more a job is held by one gender, the more the model favours that gender.

<p align="center">
  <img src="results/figures/f1_gap_vs_share.png" width="520" alt="Each dot is a job. The more women hold a job, the more the model favours women in it; surgeon: 56 of 100 qualified women shortlisted against 76 of 100 men; nurse: 84 of 100 women against 73 of 100 men.">
</p>

## Why it matters

A job platform that ranks candidates with a model is a high-risk AI system under the EU AI Act (Annex III). The provider has to examine possible bias, mitigate it and document it. Accuracy alone cannot show where a bias comes from; explanations can point at the words responsible and test whether removing them helps.

Explainable-AI methods answer why. Pronouns push scores towards the gender that usually holds a job. Other words that go with gender, such as a website name found only in women's bios, do the same. Gender is also still readable inside the model after the pronouns are removed, so removing them narrows the gap by about a fifth, not more.

**How to read this.**

- The [report](report/main.pdf) has the full argument. Each finding here starts with the result in one sentence. The numbers behind it, with 95% intervals, are in [docs/results.md](docs/results.md).
- A candidate is *qualified* when the bio belongs to someone who really does the job. The model *shortlists* a bio when it labels it with the job of the search, so the recruiter sees it.

## Key findings

- **Gender words and proxy words push the scores ([F3](docs/results.md#f3-which-words-make-up-the-gap-between-women-and-men)).** In surgeon bios, *she* and *her* lower the model's score for "surgeon" and *he* and *his* raise it, so in sum gender words account for a lower score for women than for men; nurse bios show the reverse. A proxy word is a word that is not a gender word but that mostly one gender's bios contain. For the job "fashion model" the strongest single word is *freeones*, the name of a website listing adult performers. It is in 54 of 150 women's bios and in none of the 150 men's. It may reflect the work or where the bio was published; I cannot tell which, and either way the model uses it as a gendered shortcut.
- **Swapping *she* for *he* changes few decisions ([F4](docs/results.md#f4-does-changing-only-the-gender-change-the-decision)).** Gender words push the scores, yet the swap changes the decision for 7 of every 100 bios, about as often as deleting the pronouns does (6 of 100). The decisions that do change lean towards the job that the new gender typically holds. The simpler baseline model reacts more: 14 against 9 of every 100.
- **The explanations hold in aggregate and depend on the method word by word ([F5](docs/results.md#f5-can-the-explanations-be-trusted)).** Removing the words an explanation ranks highest hurts the model far more than removing random words, so the explanations point at what matters. The exact order of the words in one bio changes with how the explanation is computed, so the conclusions rest on totals over hundreds of bios.
- **The model keeps gender after the pronouns go ([F6](docs/results.md#f6-gender-inside-the-model)).** Gender is carried by the words themselves, such as names, topics and website names, and the model does not remove it: its last layer holds about as much as word counts on the same text. A linear probe guesses gender from the main model's internal states for 58% to 63% of bios, against 52% from the job alone. The effect of swapping pronouns moves from the swapped words to the other positions of the bio within a few layers.
- **Removing the pronouns narrows the gap by a fifth and leaves most of it ([F7](docs/results.md#f7-does-removing-the-cues-close-the-gap)).** Training and scoring on bios without pronouns brings the shortlist gap from 0.143 to 0.115 for half a point of macro-F1, both measured on the test bios outside the showcase subset. That is a 20% cut (8% to 30%) from one training run per model; scrubbing only at scoring time, with no retraining, cuts it by 17% (8% to 25%). What remains is still more than twice the 95th percentile of what chance gives (0.050). I recommend it as a first step, not as a fix: in the job "fashion model" the model still shortlists 40 more qualified women than men per 100 of each, against 44 before. Deleting the six proxy words barely moves it.
- **A local LLM screener shows no gap of 5 points or more between women's and men's names, but having a name at all changes its answers ([F8](docs/results.md#f8-does-an-llm-screener-react-to-names)).** With a woman's name instead of a man's, its yes-rate for qualified candidates differs by -0.5 points (interval -4.7 to +3.5). Names from Spain, Morocco, Romania, Hispanic America and the United Kingdom show no difference this data can detect. Any name lowers the yes-rate by about 6 points against no name. That is a lead, not a finding: some bios already contain a name, and adding one sometimes changes the pronouns. Rewording the instruction changes 13% and 16% of its decisions.
- **A more accurate model is not a fairer one.** The transformer is right more often than a simple word-count baseline (84 against 80 of every 100 bios), but its shortlist gap is not clearly smaller.

## Limits and intended use

This is a research audit. Do not use these models to screen, rank or reject real people.

- Each model has one training run, so a difference between two models includes run-to-run variation I did not measure.
- The data are short web bios and the task is predicting an occupation. That is a proxy for screening, not a real hiring decision, and the findings carry to CV screening only as far as that proxy holds. Gender is binary and inferred from pronouns, so the audit says nothing about non-binary people.
- The LLM audit covers one small model, one prompt and six jobs. A small gap on this data does not show that a screener is fair.
- Removing pronouns is a first step. It does not make the screener fair.

More in the [model card](docs/model_card.md#caveats-and-limitations).

## What the audit does

| Question | Method | Kind |
|---|---|---|
| Who is shortlisted less often, and by how much? | Compare the share of qualified women and men shortlisted, with bootstrap intervals and a shuffle test | Bias metric |
| Which words drive one decision? | Word-level SHAP: how much each word adds to the score | Local (one decision) |
| Which words make up the gap between women and men? | Split the score gap exactly into one part per word | Global (many bios) |
| Does the decision change if only the gender changes? | Swap he and she, with deleting the pronouns as a control | Local and global |
| Can the explanations be trusted? | Delete words in order of importance; rerun with another seed or method | Explanation check |
| Where does the model store gender? | Read gender from each layer with a linear probe; copy states between runs (activation patching) | Mechanistic |
| Does removing the cues close the gap, and at what cost? | Retrain on text without pronouns; delete proxy words | Mitigation |
| Does a general-purpose LLM do the same with names? | Black-box audit with names from four origin groups (Spain and Hispanic America merged, because the model could not tell their names apart) | Audit |

Every result, figure and the reasoning behind each number are in **[docs/results.md](docs/results.md)**, which also lists the script that makes each figure. Each technique has an entry in **[docs/methods.md](docs/methods.md)**, and the models and their limits are in the **[model card](docs/model_card.md)**.

## Results

Tested on 99,069 bios the models never saw, with 95% bootstrap intervals.

| Model | Right job, out of 100 bios | Macro-F1 | Shortlist gap |
|---|---|---|---|
| Word counts + logistic regression (baseline) | 79.5 [79.3, 79.8] | 0.728 [0.724, 0.733] | 0.167 [0.151, 0.190] |
| DeBERTa-v3-small, fine-tuned (main model) | **83.7** [83.5, 83.9] | **0.783** [0.779, 0.788] | 0.148 [0.134, 0.172] |

Macro-F1 averages the quality over the 28 jobs, so small jobs count as much as large ones. The true positive rate (TPR) is the share of qualified candidates the model shortlists. The shortlist gap is the root mean square (RMS) over the 28 jobs of the TPR difference between women and men: 0 would be equal treatment, and shuffling the genders at random gives about 0.04. It is 0.148 on the full test set, about 15 points on a typical job, and 0.143 on the test bios outside the showcase subset, which the mitigation uses.

## Data and models

- **Data**: [Bias in Bios](https://huggingface.co/datasets/LabHC/bias_in_bios) (De-Arteaga et al., 2019, MIT licence): English short biographies (text) labelled with one of 28 occupations and a binary gender inferred from pronouns. Details in the [data card](docs/data_card.md).
- **Models**: a TF-IDF plus logistic regression baseline, and DeBERTa-v3-small fine-tuned on a 60,000-bio subsample (the main model). Explanations run on a fixed showcase subset of 1,721 test bios from six occupations: surgeon, software engineer and composer (mostly men), dietitian, nurse and model (mostly women).
- **Decisions** are recorded in [docs/decisions/](docs/decisions/); terms are defined in the [glossary](GLOSSARY.md).

## Repository layout

```text
src/hiring_bias_xai/   library: data, models, fairness metrics, attribution, plotting
report/                the report (main.pdf, LaTeX source and bibliography)
prompts/               the job ads and the prompt of the LLM audit
scripts/               one entry point per experiment; each writes to results/
results/               metrics (JSON, CSV) and figures (PDF, PNG)
notebooks/audit.ipynb  results viewer: the audit told from the saved results; CI runs it
docs/model_card.md     the three screeners and the LLM: recipe, metrics, limits
docs/methods.md        one entry per technique: what it does, how to read it, pitfalls
tests/                 pytest suite, run by CI on every pull request
```

## How to run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The lock file pins PyTorch to the CUDA 12.8 build. Run the steps in this order; times are measured on the RTX 3050 Laptop GPU (4 GB). Experiments store their inputs and intermediates outside the repository, in `HBX_ARTIFACTS` (default `~/.cache/hiring-bias-xai`). This directory holds downloaded data and pretrained weights (`hf/`), trained models (`models/`), name lists (`ine/` and `names/`), test predictions (`predictions/`), per-bio Shapley values (`attributions/`) and probe inputs and partial results (`probing/`). Each command is `uv run python scripts/<name>.py`, with the arguments listed in [docs/results.md](docs/results.md#reproduce-every-result). Without `uv`: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128`, then `pip install -e . --no-deps`.

| Step | Commands, in order | Time |
|---|---|---|
| Set up | `uv sync`; `uv run pytest -m "not gpu and not slow"` | tests about 3 minutes |
| Data | `scripts/prepare_data.py` (pinned revision) | – |
| Models | `scripts/train_baseline.py`; `scripts/train_deberta.py` | 5 minutes on the CPU; 3.45 hours on the GPU for two learning rates |
| F1 | `scripts/gap_statistics.py`; `scripts/figure_f1.py` | – |
| F2, F3 | `scripts/shapley_pilot.py`; `scripts/shapley_run.py`; `scripts/shapley_stability.py`; `scripts/word_units.py`; `scripts/figure_f2.py`; `scripts/gap_decomposition.py`; `scripts/figure_f3.py` | run 1.97 hours for the main model and 5 minutes for the baseline; decomposition about 30 seconds |
| F4 | `scripts/fetch_first_names.py`; `scripts/counterfactual_hand_check.py`; `scripts/counterfactual_gender.py`; `scripts/figure_f4.py` | scoring about 45 seconds |
| F5 | `scripts/deletion_curves.py`; `scripts/stability_runs.py`; `scripts/stability.py`; `scripts/figure_f5.py` | curves about 6 minutes; the seed and removal reruns about 10 and 15 minutes |
| F6 | `scripts/probing_features.py`; `scripts/probing.py`; `scripts/patching.py`; `scripts/figure_f6.py` | features about 1 minute each; probes 1, 4 and 5 minutes; patching 32 seconds |
| F7 | `scripts/proxy_words.py`; `scripts/train_scrubbed.py`; `scripts/mitigation.py`; `scripts/figure_f7.py` | training about 1.9 hours; scoring about 2.8 minutes per condition and 2 minutes for the report |
| F8 | `scripts/llm_names.py`; `scripts/llm_name_check.py`; `scripts/llm_pilot.py`; `scripts/llm_run.py`; `scripts/llm_analyse.py`; `scripts/figure_f8.py` | name check 3 minutes; pilot 2.4 minutes; run 2.99 hours; analysis about 25 seconds |

The arguments of each script and the order inside a step are in [docs/results.md](docs/results.md#reproduce-every-result). A dash means I did not measure the time, so only the measured times are given. The notebook is a results viewer: it loads no model, runs on a CPU in seconds, and each stage of the audit is one script (the figure-to-script table is in [docs/results.md](docs/results.md#reproduce-every-result)). Run it with `uv run pytest --nbmake notebooks/audit.ipynb`. Before a pull request, `uv run python scripts/check_docs.py` checks the links and the numbers of this page and the model card against `results/`.

For a CPU reproducibility check, use an empty `HBX_ARTIFACTS` directory and run `prepare_data.py`, `train_baseline.py` and `shapley_stability.py` in that order with `uv run python scripts/<name>.py`. Compare `results/metrics_baseline.json` (excluding `runtime_seconds`) and `results/shapley_stability.json` with the committed versions.

## Cite

- **Data.** De-Arteaga, M. et al. (2019). Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting. FAT*. Version of Ravfogel et al. (2020) on Hugging Face: `LabHC/bias_in_bios`.
- **Model.** He, P., Gao, J. and Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. ICLR 2023.
- **Libraries and tools.** PyTorch (Paszke et al., 2019), Hugging Face Transformers (Wolf et al., 2020) and Datasets (Lhoest et al., 2021), scikit-learn (Pedregosa et al., 2011), spaCy (Honnibal et al., 2020), SciPy (Virtanen et al., 2020), NumPy (Harris et al., 2020), pandas (McKinney, 2010) and Matplotlib (Hunter, 2007). The Shapley attribution follows SHAP (Lundberg and Lee, 2017) but is my own word-level code. I used an AI coding assistant for implementation and editing; I designed the audit, chose the methods and checked every result.
- **Name lists.** Instituto Nacional de Estadística (INE): Padrón Continuo, 01/01/2022, and Censos de población anuales, 01/01/2025 (CC BY 4.0). Elaboración propia con datos extraídos del sitio web del INE: www.ine.es.
- **LLM audited in F8.** Prism ML, Ternary Bonsai 2 9B (`prism-ml/Ternary-Bonsai-2-9B-gguf`, Apache-2.0), a 2-bit ternary version of Qwen3.5-9B.

## License

[MIT](LICENSE). The dataset keeps its own licence.
