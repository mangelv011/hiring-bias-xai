# Methods

Every technique the audit uses, in the order the audit meets them. Each entry says what the technique does, why it fits this model and data, how to read its output, what can go wrong, and where the code and the design record are. The findings themselves are in [results.md](results.md); terms are in the [glossary](../GLOSSARY.md).

The setting throughout: the main model reads a bio and decides which of 28 jobs it describes. A recruiter searching for "surgeon" gets the bios it labels surgeon. A candidate is qualified when the bio belongs to someone who really does that job.

## Bootstrap intervals and the permutation null

**What it does.** The bootstrap redraws the test bios with replacement 1,000 times and recomputes a number each time; the middle 95% of the results is the interval. The correlation between the gap and the share of women also gets an occupation bootstrap, which redraws the 28 jobs instead of the bios. The permutation null shuffles gender among the bios of each job 1,000 times, keeps the model's decisions, and recomputes the overall gap (the RMS TPR gap). That gives the gap that chance alone produces. Holm's correction adjusts the p-values when the 28 jobs are tested one by one.

**Why it fits.** Some jobs have few bios of one gender: the test split holds 71 men dietitians. A gap measured on so few needs an interval. The shuffle gives a yardstick for the overall gap without assuming a distribution.

**How to read it.** In [F1](results.md#f1-the-gap-follows-the-share-of-women) every dot is one job's gap, the share of qualified women passed on minus the share of qualified men, with its interval. When the interval includes 0, this data cannot tell that gap from zero; that does not make it zero. The overall gap is compared with the 95th percentile of the shuffled gaps, which is about 0.05.

**Pitfalls.** The intervals cover the choice of test bios and nothing else: each model has one training run, and the Shapley values have their own sampling noise. 1,000 shuffles cannot give a p-value below 1 in 1,001. Overlapping intervals do not show that two models are equally fair.

**Script and ADR.** `scripts/gap_statistics.py`, with `src/hiring_bias_xai/fairness.py` and `gap_statistics.py`. Recorded in [ADR 0002](decisions/0002-models.md) (the intervals of both models) and [ADR 0007](decisions/0007-mitigation.md) (the shuffle and the paired intervals).

## Shapley attribution with permutation sampling

**What it does.** It splits a bio's score for its job among its words. A word's value is the amount it adds to the score, averaged over many orders in which the words could be revealed, starting from an empty bio. The values add up exactly to the score of the full bio minus the score of the empty one. I use 40 random orders per bio, each also read backwards, and "absent" means the word is deleted.

**Why it fits.** A bio has no neutral version to compare against, and DeBERTa-v3 was not pretrained with a mask token, so methods that need a blank reference misbehave. Shapley values only need the model's outputs on bios with words removed, and their exact addition is what lets the gap decomposition below work.

**How to read it.** [F2](results.md#f2-a-woman-and-a-man-and-the-words-that-moved-the-decision) shows two wrong decisions: each bar is how much one word moved the score towards the wrong job. Read single words as indications and totals over many bios as findings.

**Pitfalls.** The values come from random orders, so a rerun differs: two runs agree at a rank correlation of 0.89. The values depend on how a word is removed (see the stability checks below). Two words that carry the same information share the credit. Punctuation can glue two words into one unit, which happens to 1.2% of units.

**Script and ADR.** `scripts/shapley_pilot.py` (sets the removal and the number of orders), `scripts/shapley_run.py` (the run) and `scripts/shapley_stability.py` (agreement of two seeds on the baseline model). Recorded in [ADR 0003](decisions/0003-shapley-attribution.md).

## Gap decomposition

**What it does.** For one job it takes the main model's average score for that job in women's bios minus the average in men's bios. Because Shapley values add up exactly, this gap splits into one part per word: how much more, or less, that word adds in women's bios than in men's, counting 0 where the word is missing. Each part also splits into usage and weight: how often the genders use the word, and how differently the model weighs it.

**Why it fits.** The question is why the screener passes on women and men at different rates. The decomposition answers it in words, and the sum of the parts is checked against the gap to float precision.

**How to read it.** [F3](results.md#f3-which-words-make-up-the-gap-between-women-and-men) lists the ten words with the largest part for two jobs, gender words in the highlight colour. The gender words' total is the part of the gap that pronouns and titles account for. Shares can pass 100% when other words pull the other way.

**Pitfalls.** It describes how the score is made up, not what happens when a word is edited; the counterfactual swap tests that. A share of the gap is unstable when the gap itself may be zero, so no share is given then. The intervals resample bios and leave out the Shapley sampling noise. Large parts can come from where a bio was published (the `freeones` website name), not from the person.

**Script and ADR.** `scripts/gap_decomposition.py`, with `src/hiring_bias_xai/gap_decomposition.py`. Recorded in [ADR 0004](decisions/0004-gap-decomposition.md).

## Counterfactual pronoun swaps with the deletion control

**What it does.** It makes a copy of each bio with the gender swapped (he and she, him and her, his and hers, Mr and Ms), asks the model again, and counts how often the decision changes (the flip rate). A second copy has the same words deleted instead of swapped.

**Why it fits.** Changing only the gender and watching the decision is the most direct test of whether the model uses it. The bios carry pronouns rather than names, so the edit is a pronoun swap.

**How to read it.** [F4](results.md#f4-does-changing-only-the-gender-change-the-decision) puts the swap's flip rate next to the deletion's. When they match, editing the text explains the flips; when the swap flips more, the model reacts to the gender itself. The share of flips that move towards a job typical of the new gender is read the same way.

**Pitfalls.** The swap leaves words such as *husband* and first names in the text, so some bios are flagged and the result is also given without them. A tagger decides whether *her* is *his* or *him*, and 3 of 50 swapped bios I read read wrongly. A swap between two genders says nothing about non-binary people.

**Script and ADR.** `scripts/counterfactual_gender.py`, with `scripts/counterfactual_hand_check.py` for the 50 bios read by hand and `src/hiring_bias_xai/counterfactual.py` for the swap. Recorded in [ADR 0005](decisions/0005-counterfactual-gender-swap.md).

## Deletion curves and stability checks

**What it does.** A deletion curve removes a bio's words step by step, up to half of them, in three orders: highest Shapley value first (MoRF), lowest first (LeRF) and at random. It follows how far the score of the true job falls. The stability checks compare the ranking of the main run with a rerun under another seed, with `[UNK]` in place of deleted words, and with leave-one-out (remove one word at a time).

**Why it fits.** Everything above rests on Shapley values, so I first test whether they point at words the model really relies on and whether the ranking survives reasonable changes.

**How to read it.** [F5](results.md#f5-can-the-explanations-be-trusted) panel a: removing the top words first should hurt most and removing the least useful first should hurt least. Panel b: the rank correlation and the shared top-10 words between two ways of computing the same thing; 1 means the same order.

**Pitfalls.** The check removes words with `[UNK]`, which the baseline model's word counts read almost like a deletion, so for that model it partly repeats the Shapley run. Weight randomization and retraining-based checks are not run. The stability comparisons use 100 bios for two of the three alternatives.

**Script and ADR.** `scripts/deletion_curves.py`, `scripts/stability_runs.py` (the reruns) and `scripts/stability.py` (leave-one-out and the comparison). Recorded in [ADR 0003](decisions/0003-shapley-attribution.md).

## Linear probing

**What it does.** A probe is a logistic regression that tries to guess gender from the model's internal representation of a bio at one layer. It learns on 10,000 training bios without pronouns and is scored on the 1,721 showcase bios.

**Why it fits.** Scrubbing the pronouns is the usual first fix, so the question is whether gender is still in the model afterwards. A linear probe keeps the claim modest: what a straight line can read, a later layer can read too.

**How to read it.** [F6](results.md#f6-gender-inside-the-model) panel a: the probe's accuracy per layer, against three guesses made from the same bios (the commoner gender, the gender typical of the job, and word counts). Above all three, the layer holds gender beyond what the job or the words alone give. A permutation test that shuffles gender within each job gives the chance level.

**Pitfalls.** A probe that reads gender shows the information is there, not that the model uses it. The shuffle keeps part of the real signal, so the null scores above the job-only guess. The probe is trained on 28 jobs and scored on six. A nonlinear probe is not run.

**Script and ADR.** `scripts/probing_features.py` (the representations) and `scripts/probing.py` (the probes). Recorded in [ADR 0006](decisions/0006-probing-and-patching.md).

## Activation patching

**What it does.** It takes a bio and its gender swap, runs the swapped bio, and copies in the model's internal state from the original bio at one layer, at the swapped words only and then at all the other words. The restored share is how much of the original score comes back: 100% is all of it.

**Why it fits.** A probe is descriptive. Patching is causal: it shows where the effect of the swap sits and how fast it moves away from the swapped words.

**How to read it.** [F6](results.md#f6-gender-inside-the-model) panel b: two curves over the layers. Early on the swapped words carry the effect; by layer 5 the other words do. The two ends are fixed by the model's design (embeddings and last layer) and are not findings.

**Pitfalls.** Patching shows that the states are sufficient to restore the effect, not that they are needed. It covers only the 168 bios the swap moves most, and only the main model. Pairs whose swap changes the number of tokens are dropped. The reverse direction is not run.

**Script and ADR.** `scripts/patching.py`. Recorded in [ADR 0006](decisions/0006-probing-and-patching.md).

## Scrubbed retraining and proxy-word masking

**What it does.** Scrubbed retraining trains a second model with the main model's recipe on bios without pronouns and titles. Proxy-word masking deletes, at scoring time, the few non-gendered words that one gender uses much more than the other and that push the gap the way pronouns do. Five mitigation conditions pair a model with a test text, and each is scored on the same bios.

**Why it fits.** The decomposition says the gap has two parts, pronouns and other words, and each fix acts on one of them.

**How to read it.** [F7](results.md#f7-does-removing-the-cues-close-the-gap) shows the gap and macro-F1 of each condition against the original model, with the band that shuffled genders give. The change from the original is computed on the same resampled bios, which is why its interval is narrow.

**Pitfalls.** The proxy words were chosen on the showcase subset, so the conditions are scored on the test bios outside it; that leaves dietitian without a measurable gap. Each model has one training run, so the difference between the original and the scrubbed model includes run-to-run variation. Only six words qualified. The first version of the rule picked words such as *the* and *and*; I corrected it before any evaluation, and the ADR keeps the first list.

**Script and ADR.** `scripts/train_scrubbed.py`, `scripts/proxy_words.py` and `scripts/mitigation.py`. Recorded in [ADR 0007](decisions/0007-mitigation.md).

## Black-box name audit with manipulation check and noise floor

**What it does.** A local LLM gets a job ad and a bio and answers yes or no. The same bio goes in with no header, a neutral ID line, a name from one of four origin groups and either gender, or its pronouns swapped. The difference in the yes-rate for qualified candidates is the effect. A manipulation check asks the model where each bare name comes from, to see whether a name carries the signal it is meant to. The noise floor repeats the bio alone, once identically and once with a second space after each full stop.

**Why it fits.** I cannot open this screener, so I test it from outside, as a company that buys it would. Origin cannot be read from the bios, so names are the way in.

**How to read it.** [F8](results.md#f8-does-an-llm-screener-react-to-names) shows each arm's change in the yes-rate from the no-header arm, against a grey band for the noise floor. An arm inside the band has not been shown to differ from rewording noise. The model also writes a reason with each answer; the audit counts how often the reasons of changed decisions mention name, origin or gender.

**Pitfalls.** One model, one prompt, six jobs. The manipulation check found the model cannot tell Spanish from Hispanic American names, so the two groups are merged. The pooled comparison of all names with no header was added after seeing the arms, so it is a lead. The 50 reasons were labelled by one reader, who also wrote the keyword rule. A null result here means gaps of 5 points or more are ruled out, not smaller ones.

**Script and ADR.** `scripts/llm_names.py`, `scripts/llm_name_check.py`, `scripts/llm_pilot.py`, `scripts/llm_run.py` and `scripts/llm_analyse.py`. Recorded in [ADR 0008](decisions/0008-llm-audit.md).
