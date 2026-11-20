# 0007. Mitigation: scrubbed retraining and proxy-word masking

Status: proposed. The rule and the expected outcomes were written before the proxy-word list was produced and before any condition was evaluated. Implemented in `scripts/train_scrubbed.py`, `src/hiring_bias_xai/proxy_words.py` and `scripts/proxy_words.py`; the evaluation and F7 are added when the evaluation has run.

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

Written before the evaluation. Reference: the main model on the full test split has macro-F1 0.783 and an RMS TPR gap of 0.148 (null mean 0.036, 95th percentile 0.049).

- **(b) and (d), scrubbing.** Macro-F1 falls by less than 1 point. The RMS TPR gap falls by 10% to 40% of 0.148, because pronouns carry only part of the gap (ADR 0004, gendered sums). [medium]
- **(c).** Close to (d) in macro-F1 and gap. The scrubbed model never saw a pronoun, so raw text adds little. [medium]
- **(e), masking.** Macro-F1 falls by less than 0.5 point. The gap falls by less than scrubbing does, and its interval overlaps the one of (a). [low]
- **No condition reaches the null.** The RMS TPR gap stays above the 95th percentile of the null in all five. [high]
- **Order of the cost.** None of the conditions loses more than 2 points of macro-F1. [medium]

## Recommendation

Added after the evaluation, with the numbers.

## Alternatives not run

- **LEACE** (Belrose et al. 2023). A closed-form linear map that removes everything a linear classifier can read about gender from the representation before the head. It needs no retraining and has a guarantee for linear readers. Not run: concept erasure is out of scope for this issue. The probes of ADR 0006 show gender is readable after scrubbing, which makes it the natural next step.
- **INLP** (Ravfogel et al. 2020). The same goal by repeated projection, one linear probe at a time. Slower, and it gives no guarantee at the end. Not run for the same reason.
- **Adversarial training.** Out of scope.
- **Choosing the proxy words on a split that is not part of test.** It would need Shapley values on new bios, about 2 hours of GPU per model. Evaluating on test minus the showcase subset keeps the choice and the measurement apart at no cost.
