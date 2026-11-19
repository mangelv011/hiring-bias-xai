# hiring-bias-xai

Terms used identically in code, tests, figures and the report.

## Task and data

**Occupation**:
One of the 28 labels of Bias in Bios; the job a bio belongs to.
_Avoid_: job, class, category

**Screening decision**:
The model's yes/no call on whether a bio matches the occupation a recruiter searches for.
_Avoid_: prediction, hiring decision

**Protected attribute**:
A property of the person the screening decision must not depend on: gender or origin group.
_Avoid_: sensitive feature, demographic

**Origin group**:
One of four groups signalled by a name: Spain, Morocco, Romania, Hispanic America (Colombia, Ecuador and Venezuela).
_Avoid_: ethnicity, nationality, Latin America

**Showcase occupations**:
The six occupations used for illustrative figures: the three with the lowest and the three with the highest share of women in the training subsample, among those with at least 500 test bios.
_Avoid_: example occupations

**Bio ID**:
The identifier of a bio, `<split>-<row index in the official split>`, such as `test-1234`. The dataset has no ID column.
_Avoid_: row number, index

**Training subsample**:
The 60,000 train bios (10,000 for dev) drawn at random within each occupation and gender group, in proportion to its size. The test split is never subsampled.
_Avoid_: sample, subset

**XAI subset**:
The fixed 2,000 test bios, stratified by occupation and gender, on which all expensive explanations are computed.
_Avoid_: sample, explanation set

## Models

**Classifier**:
Anything that maps bios to occupation probabilities, with `labels` and `predict_proba`; the baseline model, the main model and their mitigated variants satisfy it, the local LLM does not.
_Avoid_: model (when the interface is meant), predictor

**Baseline model**:
The TF-IDF plus logistic regression classifier that the transformer is compared against.
_Avoid_: baseline (unqualified)

**Main model**:
The fine-tuned DeBERTa-v3-small classifier; the one whose decisions and internals the audit explains.
_Avoid_: transformer, DeBERTa (unqualified), the model

**Checkpoint**:
The saved weights and tokenizer of the main model at its best dev macro-F1, kept in the artifacts directory.
_Avoid_: snapshot, saved model

**Representation**:
A vector the main model builds for a bio: the [CLS] state after the embeddings and after each layer, and the final vector its classification head reads. Probes work on these.
_Avoid_: embedding, features, hidden states (unqualified)

**Screener**:
A classifier viewed as the party making screening decisions, including the local LLM.
_Avoid_: judge

## Bias measurement

**TPR gap**:
TPR of women minus TPR of men within one occupation; positive favours women.
_Avoid_: equal opportunity difference, gender gap

**RMS TPR gap**:
Root mean square of the TPR gaps over all occupations; one number for how far a screener is from equal opportunity, whichever gender it favours.
_Avoid_: average gap, mean gap

**Gap-share correlation**:
Pearson correlation, across occupations, between the TPR gap and the share of women in the training subsample.
_Avoid_: bias correlation

**Bootstrap CI**:
The 95% interval of a metric from resampling the test bios with replacement and reading the 2.5th and 97.5th percentiles.
_Avoid_: error bar, confidence band

**Counterfactual swap**:
An edit of a bio that changes only the protected attribute signal (pronouns or a name header), used to see whether the screening decision flips.
_Avoid_: perturbation, augmentation

**Flip rate**:
Share of bios whose screening decision changes under a counterfactual swap.
_Avoid_: sensitivity

## Explanation

**Attribution**:
A score per input word of how much it moved the model's output.
_Avoid_: importance, saliency

**Reference input**:
The neutral input (such as `[PAD]` or `[MASK]`) that Integrated Gradients compares a bio against.
_Avoid_: baseline (unqualified)

**Scrubbing**:
Removing gender words from a bio so that only indirect signal remains.
_Avoid_: anonymizing, masking

**Probe**:
A simple classifier trained on a model's internal representations to read off the protected attribute.
_Avoid_: detector

**Control task**:
The same probe trained on random labels, to separate what the representation encodes from what the probe can memorise.
_Avoid_: random baseline

**Concept erasure**:
Removing the linear direction that encodes a protected attribute from a representation (LEACE).
_Avoid_: debiasing

**Deletion curve**:
Model confidence as words are masked in order of attribution; a faithful attribution makes it fall faster than a random order.
_Avoid_: ablation curve
