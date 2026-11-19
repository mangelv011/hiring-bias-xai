# 0002. Models: linear baseline, DeBERTa-v3-small, local LLM

Status: accepted

## Context

The audit compares what different screeners do and explains why. Explanations need different access: gradients for Integrated Gradients and probing, weights for a linear reading, only input and output for a black box. Training must fit a laptop with a 4 GB GPU.

## Decision

Three screeners. The first two satisfy the `Classifier` interface; the LLM answers yes/no and is queried directly.

- **Baseline model**: TF-IDF (1-2 grams) with logistic regression. Its coefficients can be read directly, and it anchors the comparison.
- **Main model**: DeBERTa-v3-small fine-tuned on a stratified subsample of about 60k bios. It supplies gradients and hidden states.
- **Black-box screener**: a local LLM served through an OpenAI-compatible API, used with counterfactual swaps only.

## Alternatives considered

- **Larger encoders** (DeBERTa-v3-base or large, RoBERTa-large). They do not fit 4 GB of VRAM or the time budget for fine-tuning and for the attribution runs.
- **Only an LLM.** No gradients, so no Integrated Gradients and no probing.
- **Only the linear model.** Nothing to explain mechanistically; it cannot show what a pretrained transformer encodes about gender.

## Consequences

- The subsample caps accuracy below what the full 396k bios would give. The comparison across models still holds because all are scored on the same test split.
- Expensive explanations run on the XAI subset, not on the whole test set.
- A new classifier, such as a mitigated variant, only needs `labels` and `predict_proba`; fairness and counterfactual code do not change.
