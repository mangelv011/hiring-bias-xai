# hiring-bias-xai

Terms used identically in code, tests, figures and the report.

## Task and data

**Occupation**:
One of the 28 labels of Bias in Bios; the job a bio belongs to.
_Avoid_: class, category (in prose "job" is the plain word and means the same)

**Screening decision**:
The model's yes/no call on whether a bio matches the occupation a recruiter searches for.
_Avoid_: prediction, hiring decision

**Protected attribute**:
A property of the person the screening decision must not depend on: gender or origin group.
_Avoid_: sensitive feature, demographic

**Origin group**:
One of five groups signalled by a name: Spain, Morocco, Romania, Hispanic America (Colombia, Ecuador and Venezuela) and the United Kingdom, which is the reference group. The name check can merge or drop groups the model cannot tell apart.
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

**Scrubbed model**:
The main model's recipe trained on scrubbed bios, with the same learning rate, epochs and seed. Saved as the checkpoint `deberta-scrubbed`. The main model is its counterpart, the original model, when the two are compared.
_Avoid_: debiased model, fair model

**Checkpoint**:
The saved weights and tokenizer of a DeBERTa model at its best dev macro-F1, kept in the artifacts directory under a name: `deberta` for the main model, `deberta-scrubbed` for the scrubbed model.
_Avoid_: snapshot, saved model

**Representation**:
A vector the main model builds for a bio: after the embeddings and after each layer, the [CLS] state or the mean of the token states over the non-padding tokens, and the final vector its classification head reads. Probes work on the mean.
_Avoid_: embedding, features, hidden states (unqualified)

**Screener**:
A classifier viewed as the party making screening decisions, including the local LLM.
_Avoid_: judge

## LLM audit

**Name header**:
The first line `Name: <first name> <surname>` added above a bio, with the pronouns of the bio set to match the gender of the name. The name comes from the INE lists of an origin group.
_Avoid_: name swap (that term is for editing a name inside a text)

**Arm**:
One way of presenting the same bio to the LLM: no header, neutral header (`Candidate ID: <n>`), name header for one origin group and gender, or gender swap with no header.
_Avoid_: condition, variant

**Manipulation check**:
Asking the LLM for the origin and gender of each bare full name, to see whether a name header carries the signal it is meant to carry. Groups it cannot tell apart are merged or dropped by the rule of ADR 0008.
_Avoid_: sanity check, validation

**Profile**:
One candidate of the LLM audit: a showcase bio with no first name in its text, drawn as qualified or not qualified for the job ad of one showcase occupation. Every profile is shown in every arm.
_Avoid_: case, sample

**Yes-rate**:
The share of the LLM's answers that are yes. For qualified candidates it is the TPR of the LLM; for the others it is the false positive rate.
_Avoid_: acceptance rate, hit rate

**Merge rule**:
The rule of ADR 0008, fixed before the main run, that joins the origin groups whose names the LLM confuses in the manipulation check. It joined Spain and Hispanic America.
_Avoid_: clustering

**Qualified candidate**:
A candidate whose bio really is of the job in the ad. The TPR of the LLM is the share of qualified candidates it says yes to.
_Avoid_: positive, good candidate

**Noise floor**:
How much the LLM's decisions and TPR change when nothing that matters changes: an identical rerun and a whitespace edit. An effect smaller than the floor is not read as an effect.
_Avoid_: baseline noise, error margin

**Stated reason**:
The `reason` the LLM writes before its yes or no. It is read to see whether the attribute that changed the decision is named.
_Avoid_: explanation, justification

**Keyword rule**:
A reason mentions the attribute when it has, as a whole word, one of the words listed in `llm_metrics.ATTRIBUTE_WORDS` (name, origin, nationality, gender and their kin). A first name used only to refer to the candidate is not a mention. Its agreement with hand labels is reported.
_Avoid_: detector

**Primary contrast**:
The comparison fixed before the main run of the LLM audit: the yes-rate for qualified candidates with a woman's name minus that with a man's name, paired by bio and origin group, pooled over the six jobs.
_Avoid_: main effect

## Bias measurement

**TPR gap**:
TPR of women minus TPR of men within one occupation; positive favours women.
_Avoid_: equal opportunity difference

**RMS TPR gap**:
Root mean square of the TPR gaps over all occupations; one number for how far a screener is from equal opportunity, whichever gender it favours.
_Avoid_: average gap, mean gap

**Gender gap**:
The plain-language name of the RMS TPR gap, used in the README, the figures and the prose of the results: how far, over all the jobs, the share of qualified women passed on is from the share of qualified men. 0 would be equal treatment.
_Avoid_: bias score, fairness score

**Gap-share correlation**:
Pearson correlation, across occupations, between the TPR gap and the share of women in the training subsample.
_Avoid_: bias correlation

**Bootstrap interval**:
The 95% interval of a metric from resampling the test bios with replacement and reading the 2.5th and 97.5th percentiles.
_Avoid_: error bar, confidence band, CI (unqualified)

**Occupation bootstrap**:
The 95% interval of a statistic across occupations, from resampling the occupations (the 28 gap and share points) with replacement. It answers how much the gap-share correlation depends on which occupations exist; the bootstrap interval over bios does not.
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
Share of bios whose screening decision changes under a counterfactual swap or under the removal control.
_Avoid_: sensitivity

**Gender swap**:
The counterfactual swap of gendered pronouns and titles in a bio (he for she, his for her, Mr for Ms), keeping capitalisation. The words listed as residual cues are not swapped.
_Avoid_: pronoun flip, gender flip

**Removal control**:
The same bios with the gendered pronouns and titles deleted (`data.scrub`) instead of swapped. It separates the effect of gender from the effect of editing the text.
_Avoid_: ablation, scrubbed swap

**Residual cue**:
A word that still shows the gender of the person after a gender swap: a gendered noun from the list fixed in `counterfactual.RESIDUAL_NOUNS` (mother, husband, actress...) or a first name from the SSA list used as a person's name in the bio.
_Avoid_: leak, gendered word (that term is the scrubbing list)

**Flagged bio**:
A bio with at least one residual cue. Results are reported for all bios and for the unflagged ones.
_Avoid_: contaminated bio

**Flip direction**:
For a flipped bio under the gender swap, whether the new prediction has a training share of women that moves toward the new gender: lower for a woman's bio turned male, higher for a man's bio turned female. It uses the same share as the gap-share correlation, so the two results are not independent.
_Avoid_: bias direction

**Hand check**:
The 50 swapped bios drawn at random and read one by one for grammar and meaning; the share judged incoherent is reported with its Wilson interval.
_Avoid_: manual validation

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
_Avoid_: anonymizing, masking (a different step: see proxy-word masking)

**Probe**:
A logistic regression on standardised representations, trained to read gender from one layer of an encoder. Its accuracy on the showcase subset is compared with references (the majority class, the occupation alone, TF-IDF) and with a permutation test that shuffles gender within occupation and refits it. A probe that reads gender does not show that the model uses it.
_Avoid_: detector

**Activation patching**:
Running the swapped bio (corrupted) while replacing the hidden states of one layer output with those of the original bio (clean), at chosen token positions, to see how much of the clean score returns. This is denoising: it shows that those states are sufficient, not that they are necessary.
_Avoid_: causal tracing, ablation

**Eligible bio**:
A bio whose gender swap moves the log-odds score of the true occupation by more than the 95th percentile of the move under the removal control. Patching uses only these.
_Avoid_: selected bio

**Restored share**:
For one bio and one layer output, (patched - corrupted) / (clean - corrupted) of the log-odds score of the true occupation. 1 means the patch brings the clean score back, 0 that it changes nothing. It is reported for bios whose swap moves the score by more than the 95th percentile of the move under the removal control.
_Avoid_: recovery, patching effect

**Deletion curve**:
For one screener, how its call and its log-odds score change as more words of a bio are removed in a fixed order, from 0% to 50% of the words. It is summarised by its area: the mean over the steps. An attribution is faithful if removing its highest words hurts more than removing random ones.
_Avoid_: ablation curve, confidence curve

**MoRF**:
Most relevant first: the deletion order that removes the words with the largest signed Shapley value for the true occupation first.
_Avoid_: top-first order

**LeRF**:
Least relevant first: the deletion order that removes the words with the smallest signed Shapley value for the true occupation first. A faithful attribution makes it hurt less than a random order.
_Avoid_: bottom-first order

**Leave-one-out occlusion**:
The value of a word is the log-odds score of the full bio minus the score with that word deleted, the others kept. It is the cheap comparison for Shapley values; two words that carry the same information each get a small value, because the other covers for it.
_Avoid_: occlusion (unqualified), ablation

**Top-10 overlap**:
For two attributions of the same bio, the share of the ten words with the largest absolute value in one that are also among the ten largest in the other. It is not the top-10 share of the gap decomposition.
_Avoid_: top-k accuracy

## Gap decomposition

**Score gap**:
In one occupation, the mean log-odds score of the full bios of women minus that of men, for the true occupation. Positive favours women. It is not the TPR gap: it uses scores, not decisions.
_Avoid_: bias score

**Word**:
In the gap decomposition, a word unit with punctuation at its ends removed and lower-cased (`data.word_core`); a unit with no letter or digit keeps its text. A bio's value for a word is the sum of the Shapley values of its occurrences; a word the bio lacks counts 0.
_Avoid_: token, term

**Word importance**:
The mean Shapley value of a word over the bios of one occupation that contain it, reported only for words in at least 20 of them.
_Avoid_: feature importance, global importance

**Gap contribution**:
The mean Shapley value of a word over the women's bios of an occupation minus the mean over the men's bios, a missing word counting 0. The contributions of all words add up to the score gap.
_Avoid_: bias of a word, word effect

**Top-10 share**:
The sum of the gap contributions of the ten words with the largest absolute contribution, divided by the score gap. It can exceed 100% when the other words pull the other way. It is unstable when the interval of the score gap includes 0.
_Avoid_: explained variance

**Gendered sum**:
The sum of the gap contributions of the gendered words: how far they put women's bios above (positive) or below (negative) men's. It needs no division, so it exists when the score gap is near 0.
_Avoid_: gendered effect, bias of gendered words

**Gendered share**:
The sum of the gap contributions of the gendered words, divided by the score gap.
_Avoid_: bias share

**Usage and weight**:
The split of a word's gap contribution into the difference in how often women's and men's bios contain it, and the difference in its mean value when present. They add up to the contribution.
_Avoid_: frequency effect, coefficient effect

## Mitigation

**Proxy word**:
A word that is not gendered, whose gap contribution has the sign of the occupation's gendered sum, that the gender the gap favours uses at least 3 times as often as the other gender (shares smoothed by 0.01), and that at least 5% of that gender's bios contain. The model can read gender from it. A word the model weighs heavily but both genders use alike, such as *the*, is not one.
_Avoid_: bias word, stand-in word

**Proxy-word masking**:
Deleting the proxy words from a bio before the model reads it. It needs no retraining.
_Avoid_: scrubbing (that term is for pronouns and titles)

**Mitigation condition**:
One pairing of a model and a test text, lettered a to e: (a) original model on raw bios, (b) original model on scrubbed bios, (c) scrubbed model on raw bios, (d) scrubbed model on scrubbed bios, (e) original model with proxy words deleted. All five are scored on the same test bios outside the showcase subset.
_Avoid_: scenario, arm (the LLM audit's term)
