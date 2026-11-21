# 0007. Mitigation: scrubbed retraining and proxy-word masking

Status: accepted. The first rule and the expected outcomes were written before any list was produced and before any condition was evaluated. The proxy rule was corrected once, after seeing the first list and before any evaluation (see the note of 2026-11-20). Implemented in `scripts/train_scrubbed.py`, `src/hiring_bias_xai/proxy_words.py`, `scripts/proxy_words.py`, `scripts/mitigation.py` and `scripts/figure_f7.py`; `fairness.paired_difference` gives the paired changes.

## Context

The explanations (ADR 0004) say the main model's score gap between women and men has two parts: gender words (pronouns and titles) and other words that go with gender in the training bios. Two mitigations follow from that, and each acts on one part. Scrubbing the pronouns and retraining targets the first. Hiding the other words at inference, with no retraining, targets the second. I run both and measure what each costs in accuracy and what it leaves of the gap.

## Decision

### Scrubbed retraining

The recipe of the main model (ADR 0002) with one change: the text. `data.scrub` is applied to the training and dev bios. Learning rate 3e-5 (the one chosen for the main model), 5 epochs, seed 42, the weights of the best dev macro-F1 restored. The checkpoint is saved as `deberta-scrubbed`, so the main model is not overwritten. The training log goes to `results/training_log_scrubbed.csv`. Dev macro-F1 is measured on scrubbed dev bios, so the choice of weights uses the scrubbed text only.

### Proxy words: the rule

Fixed here, before the list was computed. The input is `results/gap_decomposition.csv`, rows of the main model, and the gender-word sums of `results/gap_explained.json`. For each of the six showcase occupations, a word is a proxy word when all of these hold:

1. It is not a gender word (`data.is_gendered_word`).
2. The 95% interval of its gap contribution excludes 0.
3. Its contribution has the sign of the gender-word sum of that occupation. The word pushes the gap the same way as the pronouns do.
4. It is one of the 5 words with the largest absolute contribution that pass 1 to 3 in that occupation. Ties go to the alphabetical order.

The mask list is the union over the six occupations. It is saved with this rule in `results/proxy_words.json`. A model at inference does not know the occupation of a bio, so the whole list is applied to every bio. Masking deletes every whitespace-separated unit of the bio whose `data.word_core` is a word of the list.

The list was chosen on the showcase subset, which is part of the test split. To keep the choice apart from the measurement, the masking condition is evaluated on the test bios outside the showcase subset.

### Correction of the proxy rule (note of 2026-11-20)

The rule above is the first one, kept as written. It selected 26 words: an, and, appointment, born, center, currently, dental, dietitian, doctors, eating, for, freeones, gallery, has, include, jazz, llc, medical, ranked, registered, tech, technical, that, the, their, years. Most of them are function words (*the*, *and*, *for*) or words of the job itself (*dietitian*, *medical*, *dental*, *jazz*). They are not proxies. A proxy is a word that is not gendered but that women's and men's bios use at different rates, so the model can read gender from it. The first rule asked only for a contribution to the gap, and a word gets a contribution from how the model weighs it as well as from who uses it. Masking those words would test something else and would cost accuracy for the wrong reason.

The rule was corrected after seeing that list and before any condition was evaluated. Nothing is tuned after the evaluation. The corrected rule keeps conditions 1 to 3 and adds a use condition before the top 5:

1. It is not a gender word (`data.is_gendered_word`).
2. The 95% interval of its gap contribution excludes 0.
3. Its contribution has the sign of the gender-word sum of that occupation.
4. The gender the gap favours (women when the gender-word sum is positive, men when it is negative) uses it at least 3 times as often as the other gender: (share of that gender's bios that contain it + 0.01) / (share of the other gender's bios + 0.01) is 3 or more. The shares are `freq_female` and `freq_male` of `results/gap_decomposition.csv`. The 0.01 keeps the ratio finite for a word one gender never uses.
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

Each condition also gets its change from (a) with a 95% interval from `fairness.paired_difference`: both conditions are scored on the same resampled bios, so the interval is that of the difference and is much narrower than the two intervals of the conditions side by side.

**Dietitian has no gap outside the showcase subset.** The test split holds 71 men dietitians and the showcase subset takes every one of them. Outside it, dietitian has no male bio, so its TPR gap, and with it the RMS TPR gap over 28 occupations, is undefined. Macro-F1 and accuracy use all 97,348 bios outside the subset. The RMS TPR gap, its null and the paired change of the gap use the 27 occupations that have both genders there. The dietitian gap is reported on the full test split only. This was found when the first report returned no number; it changes where the gap is measured and nothing in the rules above.

One training seed per model. A difference between the original and the scrubbed model mixes the effect of the text with the run-to-run variation of training, which was not measured. The intervals cover the sampling of test bios only.

## Expected outcomes

Written before the evaluation. The bullets below were written for the first rule. After the correction, the bullet on masking (e) is re-read at the end of this section; the others do not depend on the list. Reference: the main model on the full test split has macro-F1 0.783 and an RMS TPR gap of 0.148 (null mean 0.036, 95th percentile 0.049).

- **(b) and (d), scrubbing.** Macro-F1 falls by less than 1 point. The RMS TPR gap falls by 10% to 40% of 0.148, because pronouns carry only part of the gap (ADR 0004, gender-word sums). [medium]
- **(c).** Close to (d) in macro-F1 and gap. The scrubbed model never saw a pronoun, so raw text adds little. [medium]
- **(e), masking.** Macro-F1 falls by less than 0.5 point. The gap falls by less than scrubbing does, and its interval overlaps the one of (a). [low]
- **No condition reaches the null.** The RMS TPR gap stays above the 95th percentile of the null in all five. [high]
- **Order of the cost.** None of the conditions loses more than 2 points of macro-F1. [medium]

Re-read for the corrected rule, before any evaluation. The corrected rule is stricter, so the list is short and each word is used mostly by one gender. It is dominated by the occupation *fashion model*, whose gap is the one carried by words such as *freeones*. The expectations on (e) become:

- Macro-F1 changes by less than 0.1 point. A handful of words is deleted. [medium]
- The RMS TPR gap changes by less than the width of its interval. The words belong to few occupations, and the RMS gap sums over 28. The TPR gap of *fashion model*, the occupation that carries most of the list, is the one place I expect a visible fall. [low]
- The first-rule expectation that masking "falls by less than 0.5 point" and "gap falls less than scrubbing" still stands as written.

## Outcomes against the expectations

Numbers are on the 97,348 test bios outside the showcase subset (`results/mitigation.json`); the original model has macro-F1 0.7835 and an RMS TPR gap of 0.1435, and the gap of shuffled genders is 0.037 on average, 0.050 at the 95th percentile.

- **(b) and (d), scrubbing.** Held. Macro-F1 fell by 0.76 points (b) and 0.48 points (d), under 1. The RMS gap fell by 0.025 (17%) and 0.028 (20%), inside the 10% to 40% range.
- **(c).** Held. Macro-F1 fell by 0.38 points and the gap by 0.033 (23%); the intervals of (c) and (d) overlap.
- **(e), first-rule expectation.** Held. Macro-F1 fell by 0.16 points, under 0.5. The gap fell by 0.0023 (interval from -0.0047 to +0.0004), less than scrubbing, and its interval overlaps that of (a).
- **No condition reaches the null.** Held. The lowest gap, 0.110 for (c), is more than twice the 95th percentile of the null; every permutation p-value is 1 in 1,001.
- **No condition loses more than 2 points.** Held. The largest loss is 0.76 points.
- **(e), corrected-rule expectations.** Two of three did not hold. Macro-F1 fell by 0.16 points [0.11, 0.21], not under 0.1. 12% of the bios contain one of the six words. The TPR gap of *fashion model* did not show a visible fall: 0.438 before and 0.423 after, inside the interval of either. The RMS gap changed by less than the width of its interval, as expected.

## Recommendation

For a stakeholder: scrub the pronouns and titles from the bios, and train and score on scrubbed text (condition d). It takes the RMS TPR gap from 0.1435 to 0.1152, a fall of 0.028 [0.012, 0.043], for 0.48 points of macro-F1 [0.17, 0.76]. Accuracy does not change (-0.02 points, interval -0.16 to +0.12). If retraining is not possible, scrubbing the bios the original model reads (condition b) gives most of the gap fall, 0.025 [0.012, 0.036], for 0.76 points of macro-F1. Conditions c and d are within noise of each other, and d leaves no pronoun for the model to meet.

Do not present this as making the screener fair. The gap after scrubbing is more than twice the upper edge of what shuffled genders give, in every condition. The gaps in the two jobs with the largest gap hardly moved: *fashion model* from +0.44 to +0.40 and surgeon from -0.22 to -0.18 (women minus men, so positive favours women). What is left is not in the pronouns. In *fashion model* the strongest single word was the name of a website (ADR 0004). Masking the six proxy words does not help: it leaves the gap where it was and costs 0.16 points of macro-F1, so I would not use it. The next step is a method that works on what the model has learned, not on the text. Beyond that, the gap per occupation should be measured on every new model before it is used.

Limits of this advice: one training seed per model, so the effect of retraining is not separated from run-to-run variation; the gap is measured on 27 of 28 occupations outside the showcase subset; the proxy words were chosen on the showcase subset and tested outside it, but there are only six.

## Alternatives not run

- **LEACE** (Belrose et al. 2023). A closed-form linear map that removes everything a linear classifier can read about gender from the representation before the head. It needs no retraining and has a guarantee for linear readers. Not run: concept erasure is out of scope for this audit. The probes of ADR 0006 show gender is readable after scrubbing, which makes it the natural next step.
- **INLP** (Ravfogel et al. 2020). The same goal by repeated projection, one linear probe at a time. Slower, and it gives no guarantee at the end. Not run for the same reason.
- **Adversarial training.** Out of scope.
- **Choosing the proxy words on a split that is not part of test.** It would need Shapley values on new bios, about 2 hours of GPU per model. Evaluating on test minus the showcase subset keeps the choice and the measurement apart at no cost.
