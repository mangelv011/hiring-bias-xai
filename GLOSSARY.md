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

**Showcase subset**:
The fixed test bios of the showcase occupations, up to 150 per occupation and gender, on which all expensive explanations are computed. A group with fewer test bios contributes all of them.
_Avoid_: sample, explanation set, XAI subset

## Models

**Classifier**:
Anything that maps bios to occupation scores: logits, probabilities, the predicted occupation and the log-odds of one occupation. The baseline model, the main model and their mitigated variants satisfy it, the local LLM does not.
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

**Occupation bootstrap**:
The 95% interval of a statistic across occupations, from resampling the occupations (the 28 gap and share points) with replacement. It answers how much the gap-share correlation depends on which occupations exist; the bootstrap CI does not.
_Avoid_: second-level bootstrap

**Permutation null**:
The RMS TPR gaps a screener would show if gender had no bearing on its hits: the gender labels are shuffled among the bios of each occupation, the screening decisions stay fixed, and the result is read 1,000 times. The permutation p-value is (1 + shuffles with an RMS TPR gap at least the observed) / (1 + shuffles).
_Avoid_: random baseline, null model

**Holm-adjusted p**:
The p-value of one occupation's TPR gap test after Holm's correction for testing all 28 occupations. Only a Holm-adjusted p below 0.05 supports the claim that an occupation's TPR gap is not zero; the raw gaps are descriptive.
_Avoid_: corrected p, significant gap (unqualified)

**Gap statistics**:
The statistics that put the TPR gaps in context: the gap-share correlation with its Fisher-z and occupation bootstrap intervals, Spearman's rho, the leave-one-occupation-out range, the RMS TPR gap against its permutation null, and the per-occupation Holm-adjusted p.
_Avoid_: bias statistics

**Counterfactual swap**:
An edit of a bio that changes only the protected attribute signal (pronouns or a name header), used to see whether the screening decision flips.
_Avoid_: perturbation, augmentation

**Flip rate**:
Share of bios whose screening decision changes under a counterfactual swap.
_Avoid_: sensitivity

## Explanation

**Attribution**:
A score per word of a bio of how much it moved the log-odds score; here, a Shapley value.
_Avoid_: importance, saliency

**Log-odds score**:
The score of a bio for one occupation: the log of the odds of that occupation against all the others, computed from the logits. Attributions explain this score.
_Avoid_: confidence, probability

**Removal**:
What "absent" means for a word: delete it, or replace it with one `[UNK]`.
_Avoid_: masking, ablation

**Coalition**:
The bio with some of its words present and the rest removed.
_Avoid_: subset, perturbation

**Shapley value**:
The average marginal effect of a word on the log-odds score over random orders in which the words are added to an empty bio. The values of a bio add up to the score of the full bio minus the score of the empty bio.
_Avoid_: importance, contribution

**Gendered word**:
A word unit of a bio whose core, without punctuation at its ends, is a word the scrubbing list removes: a gendered pronoun or title. `she,` is one; `sheriff` and `shoot...she` are not.
_Avoid_: gender marker, gender term

**Scrubbing**:
Removing gender words from a bio so that only indirect signal remains.
_Avoid_: anonymizing, masking

**Probe**:
A simple classifier trained on a model's internal representations to read off the protected attribute.
_Avoid_: detector

**Control task**:
The same probe trained on random labels, to separate what the representation encodes from what the probe can memorise.
_Avoid_: random baseline

**Deletion curve**:
Model confidence as words are masked in order of attribution; a faithful attribution makes it fall faster than a random order.
_Avoid_: ablation curve
