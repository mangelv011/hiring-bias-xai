# 0007. Mitigation: scrubbed retraining and proxy-word masking

Status: proposed. The first rule and the expected outcomes were written before any list was produced and before any condition was evaluated. The proxy rule was corrected once, after seeing the first list and before any evaluation (see the note of 2026-11-20). Implemented in `scripts/train_scrubbed.py`, `src/hiring_bias_xai/proxy_words.py` and `scripts/proxy_words.py`; the evaluation and F7 are added when the evaluation has run.

## Context

The explanations (ADR 0004) say the main model's score gap between women and men has two parts: gendered words (pronouns and titles) and other words that go with gender in the training bios. Two mitigations follow from that, and each acts on one part. Scrubbing the pronouns and retraining targets the first. Hiding the other words at inference, with no retraining, targets the second. We run both and measure what each costs in accuracy and what it leaves of the gap.

## Decision

### Scrubbed retraining

The recipe of the main model (ADR 0002) with one change: the text. `data.scrub` is applied to the training and dev bios. Learning rate 3e-5 (the one chosen for the main model), 5 epochs, seed 42, the weights of the best dev macro-F1 restored. The checkpoint is saved as `deberta-scrubbed`, so the main model is not overwritten. The training log goes to `results/training_log_scrubbed.csv`. Dev macro-F1 is measured on scrubbed dev bios, so the choice of weights uses the scrubbed text only.

### Proxy words: the rule

Fixed here, before the list was computed. The input is `results/gap_decomposition.csv`, rows of the main model, and the gendered sums of `results/gap_explained.json`. For each of the six showcase occupations, a word is a proxy word when all of these hold:

1. It is not a gendered word (`data.is_gendered_word`).
2. The 95% interval of its gap contribution excludes 0.
3. Its contribution has the sign of the gendered sum of that occupation. The word pushes the gap the same way as the pronouns do.
4. It is one of the 5 words with the largest absolute contribution that pass 1 to 3 in that occupation. Ties go to the alphabetical order.

The mask list is the union over the six occupations. It is saved with this rule in `results/proxy_words.json`. A model at inference does not know the occupation of a bio, so the whole list is applied to every bio. Masking deletes every whitespace-separated unit of the bio whose `data.word_core` is a word of the list.

The list was chosen on the showcase subset, which is part of the test split. To keep the choice apart from the measurement, the masking condition is evaluated on the test bios outside the showcase subset.

### Correction of the proxy rule (note of 2026-11-20)

The rule above is the first one, kept as written. It selected 26 words: an, and, appointment, born, center, currently, dental, dietitian, doctors, eating, for, freeones, gallery, has, include, jazz, llc, medical, ranked, registered, tech, technical, that, the, their, years. Most of them are function words (*the*, *and*, *for*) or words of the job itself (*dietitian*, *medical*, *dental*, *jazz*). They are not proxies. A proxy is a word that is not gendered but that women's and men's bios use at different rates, so the model can read gender from it. The first rule asked only for a contribution to the gap, and a word gets a contribution from how the model weighs it as well as from who uses it. Masking those words would test something else and would cost accuracy for the wrong reason.

The rule was corrected after seeing that list and before any condition was evaluated. Nothing is tuned after the evaluation. The corrected rule keeps conditions 1 to 3 and adds a use condition before the top 5:

1. It is not a gendered word (`data.is_gendered_word`).
2. The 95% interval of its gap contribution excludes 0.
3. Its contribution has the sign of the gendered sum of that occupation.
4. The gender the gap favours (women when the gendered sum is positive, men when it is negative) uses it at least 3 times as often as the other gender: (share of that gender's bios that contain it + 0.01) / (share of the other gender's bios + 0.01) is 3 or more. The shares are `freq_female` and `freq_male` of `results/gap_decomposition.csv`. The 0.01 keeps the ratio finite for a word one gender never uses.
5. At least 5% of the bios of that gender contain it. Rarer words cannot move the gap of a job much.
6. The 5 with the largest absolute contribution are kept, ties in alphabetical order. The union over the six occupations is the mask list.

An occupation may add no word. The list, the shares of every word and the occupations without words are in `results/proxy_words.json`.

### Evaluation

Five conditions, one per row of F7:

- (a) original model, raw test
- (b) original model, scrubbed test
- (c) scrubbed model, raw test
- (d) scrubbed model, scrubbed test
- (e) original model, raw test with the proxy words deleted

Every condition is scored on the same bios: the test split minus the showcase subset, so the five are comparable. (a) to (d) are also reported on the full test split, for continuity with the earlier tables. Per condition: macro-F1, accuracy, RMS TPR gap with the permutation null (gender shuffled within occupation, 1,000 shuffles, as in `scripts/gap_statistics.py`), and the TPR gap of each showcase occupation, all with 95% bootstrap intervals (`fairness.evaluate`).

One training seed per model. A difference between the original and the scrubbed model mixes the effect of the text with the run-to-run variation of training, which was not measured. The intervals cover the sampling of test bios only.

## Expected outcomes

Written before the evaluation. The bullets below were written for the first rule. After the correction, the bullet on masking (e) is re-read at the end of this section; the others do not depend on the list. Reference: the main model on the full test split has macro-F1 0.783 and an RMS TPR gap of 0.148 (null mean 0.036, 95th percentile 0.049).

- **(b) and (d), scrubbing.** Macro-F1 falls by less than 1 point. The RMS TPR gap falls by 10% to 40% of 0.148, because pronouns carry only part of the gap (ADR 0004, gendered sums). [medium]
- **(c).** Close to (d) in macro-F1 and gap. The scrubbed model never saw a pronoun, so raw text adds little. [medium]
- **(e), masking.** Macro-F1 falls by less than 0.5 point. The gap falls by less than scrubbing does, and its interval overlaps the one of (a). [low]
- **No condition reaches the null.** The RMS TPR gap stays above the 95th percentile of the null in all five. [high]
- **Order of the cost.** None of the conditions loses more than 2 points of macro-F1. [medium]

Re-read for the corrected rule, before any evaluation. The corrected rule is stricter, so the list is short and each word is used mostly by one gender. It is dominated by the occupation *model*, whose gap is the one carried by words such as *freeones*. The expectations on (e) become:

- Macro-F1 changes by less than 0.1 point. A handful of words is deleted. [medium]
- The RMS TPR gap changes by less than the width of its interval. The words belong to few occupations, and the RMS gap sums over 28. The TPR gap of *model*, the occupation that carries most of the list, is the one place I expect a visible fall. [low]
- The first-rule expectation that masking "falls by less than 0.5 point" and "gap falls less than scrubbing" still stands as written.

## Recommendation

Added after the evaluation, with the numbers.

## Alternatives not run

- **LEACE** (Belrose et al. 2023). A closed-form linear map that removes everything a linear classifier can read about gender from the representation before the head. It needs no retraining and has a guarantee for linear readers. Not run: concept erasure is out of scope for this issue. The probes of ADR 0006 show gender is readable after scrubbing, which makes it the natural next step.
- **INLP** (Ravfogel et al. 2020). The same goal by repeated projection, one linear probe at a time. Slower, and it gives no guarantee at the end. Not run for the same reason.
- **Adversarial training.** Out of scope.
- **Choosing the proxy words on a split that is not part of test.** It would need Shapley values on new bios, about 2 hours of GPU per model. Evaluating on test minus the showcase subset keeps the choice and the measurement apart at no cost.
