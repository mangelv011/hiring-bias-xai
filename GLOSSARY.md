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
The two occupations used for illustrative figures.
_Avoid_: example occupations

**XAI subset**:
The fixed, stratified subset of test bios on which all expensive explanations are computed.
_Avoid_: sample, explanation set

## Models

**Classifier**:
Anything that maps bios to occupation probabilities, with `labels` and `predict_proba`; every model and mitigated variant satisfies it.
_Avoid_: model (when the interface is meant), predictor

**Baseline model**:
The TF-IDF plus logistic regression classifier that the transformer is compared against.
_Avoid_: baseline (unqualified)

**Screener**:
A classifier viewed as the party making screening decisions, including the local LLM.
_Avoid_: judge

## Bias measurement

**TPR gap**:
TPR of women minus TPR of men within one occupation; positive favours women.
_Avoid_: equal opportunity difference, gender gap

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
A probe task with random labels, used to tell what a probe learns from what it memorises.
_Avoid_: random baseline

**Concept erasure**:
Removing the linear direction that encodes a protected attribute from a representation (LEACE).
_Avoid_: debiasing

**Deletion curve**:
Model confidence as the highest-attribution words are removed one by one; a faithful attribution makes it fall fast.
_Avoid_: ablation curve
