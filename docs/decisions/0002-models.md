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
- Expensive explanations run on the showcase subset, not on the whole test set.
- A new classifier, such as a mitigated variant, only needs `labels` and `predict_proba`; fairness and counterfactual code do not change.

## Update: how the main model was trained

Numbers are in `results/metrics_deberta.json` and `results/training_log.csv`.

**Setup.** DeBERTa-v3-small at a pinned Hugging Face revision, a plain PyTorch loop (AdamW, 6% linear warm-up then linear decay, bfloat16 autocast with float32 weights), learning rates 5e-5 and 3e-5, 5 epochs each, effective batch 32. `max_length` is 256: the 99.9th percentile of the training subsample is 197 tokens and 0.008% of bios (5 of 60,000) are cut. Every run goes through all epochs; the checkpoint is the weights of the evaluation with the best dev macro-F1 (25 evaluations per run, one every 375 steps).

**Result.** Learning rate 3e-5 won on dev with 0.7883 at the last evaluation (step 9375). Learning rate 5e-5 reached 0.7863 at steps 7875 and 9000. The two differ by 0.002, which is inside the 0.015 swing between neighbouring evaluations, so the choice of learning rate is close to a coin flip. Test macro-F1 is 0.7832 [0.7787, 0.7879] against 0.7282 [0.7235, 0.7332] for the baseline model; the intervals do not overlap, so the main model is more accurate and no investigation of a loss was needed. Train macro-F1 is 0.9286, a gap of 0.145 (the baseline's is 0.271).

**The two learning rates ended differently.** Learning rate 3e-5 had its best evaluation at the last step, so more epochs might help a little. Learning rate 5e-5 levelled off near 0.786 in epoch 5 while its dev loss rose from 0.62 to 0.70, a sign that more epochs would mostly add overfitting. I did not try longer runs: the two runs took 12,411 s together on the laptop GPU.

**Early stopping was dropped.** A run that stopped after 3 evaluations without improvement (0.6 of an epoch) halted at step 3750 of epoch 2 with a best dev macro-F1 of 0.7485, while a 3-epoch run without stopping reached 0.7824. Dev macro-F1 moves by about 0.015 between neighbouring evaluations and dev loss was still falling, so that patience stopped on noise. Logs: `train3_stopped.log` and the 3-epoch log, kept outside the repository.

**Batch 4 with accumulation 8, scoring batch 4.** The first runs used batches of 8 with accumulation 4 (peak allocated 3,120 MB of 4,096 MB, with about 3.9 GB reserved). The GPU driver then failed twice with Windows event `nvlddmkm` 153, at 15:24:58 and at 17:02:13, which ended the process. I think the cause is near-full VRAM together with the host commit that Windows reserves for it, but did not prove it. Batches of 4 with accumulation 8 keep the effective batch at 32; peak allocated memory in the final run is 2,863 MB. Scoring batches (dev, test, train) are also 4; they were 32 before.

**Weights come from the pinned Microsoft repository, which ships pickle weights only.** transformers loads them with `weights_only=True`, and the project writes its own checkpoint as safetensors. Loading the checkpoint takes no path and reads only `artifacts_dir()/models/deberta`. The fp16 weights of the repository are loaded as float32: float16 weights gave NaN at the first optimizer step.

**Bias.** RMS TPR gap is 0.1476 [0.1342, 0.1715] for the main model and 0.1666 [0.1506, 0.1895] for the baseline model. The intervals overlap, so the data do not show that either model is fairer.

## Update: attribution moved to Shapley values

The context and the alternatives above name Integrated Gradients as the explanation that needs gradients. The audit replaced it with word-level Shapley values. The reasons and the alternatives are in [ADR 0003](0003-shapley-attribution.md). The choice of models stands.

## Outcomes

All three screeners were built and audited. The baseline model reaches a test macro-F1 of 0.7282 and the main model 0.7832; the intervals do not overlap. The RMS TPR gap is 0.1666 for the baseline model and 0.1476 for the main model, and those intervals overlap, so the data does not show that either is fairer. The two learning-rate runs of the main model took 12,411 s together on the laptop GPU, and the baseline model took 297 s on the CPU. The LLM was queried only through counterfactual headers and swaps, as written (ADR 0008). A fourth screener, the scrubbed model, was added for the mitigation (ADR 0007). Integrated Gradients was replaced by Shapley values (ADR 0003). The model card in [model_card.md](../model_card.md) holds the current recipes.
