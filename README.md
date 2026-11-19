# hiring-bias-xai

Explainable-AI audit of a CV-screening text classifier for gender and origin bias.

## Idea

A recruiter searches for an occupation and a model decides which bios match. This project audits that screening decision on Bias in Bios (De-Arteaga et al., 2019), about 396k English biographies over 28 occupations with a binary gender label inferred from pronouns.

Three screeners are compared: a TF-IDF plus logistic regression baseline model, a fine-tuned DeBERTa-v3-small, and a local LLM used as a black-box screener. The audit measures where each one treats women and men differently, then uses explanation methods to ask why.

## Research question

When a screening model favours one gender in an occupation, which parts of the bio drive the difference, and does removing them close the gap?

## Approach

- **Bias metric**: the TPR gap per occupation (TPR women minus TPR men, positive favours women), and its correlation with the share of women in the occupation.
- **Attribution**: Integrated Gradients at word level, with a reference input such as `[PAD]` or `[MASK]`.
- **Counterfactual swaps**: pronoun swaps, and name headers built from Spanish INE name lists for four origin groups (Spain, Morocco, Romania, Hispanic America = Colombia, Ecuador and Venezuela).
- **Probing**: gender probes on scrubbed text, with a control task, plus LEACE concept erasure.
- **Sanity checks**: deletion curves and weight randomization.

Expensive explanations run on a fixed, stratified XAI subset of the test bios. Terms are defined in [GLOSSARY.md](GLOSSARY.md); design decisions are in [docs/decisions/](docs/decisions/).

## Status

Package layout, environment, CI, glossary, the first two decision records and the data pipeline (`python scripts/prepare_data.py`, described in [docs/data_card.md](docs/data_card.md)). No training or analysis code yet.

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

Experiment entry points will live in `scripts/` and write small JSON/CSV files and figures to `results/`.

## License

[MIT](LICENSE)
