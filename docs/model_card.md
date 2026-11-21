# Model card

Follows the sections of Mitchell et al., "Model Cards for Model Reporting" (2019). It covers the three screeners I trained and the LLM I audited. Terms follow the [glossary](../GLOSSARY.md). Every number comes from a file in [`results/`](../results/), and `scripts/check_docs.py` compares them on every pull request.

**The setting.** A recruiter searches for a job, say surgeon. A model reads each candidate's short professional bio and decides which of 28 jobs it describes. The bios it labels "surgeon" go on to the recruiter; the rest are set aside. A candidate is qualified for a search when the bio belongs to someone who really does that job.

## Model details

Three screeners, all developed by me for this audit and described in [ADR 0002](decisions/0002-models.md). Each gives a score for every job and decides for the job with the highest score. They share one training split and one test split. The code is under the MIT licence; the weights are not in the repository but in the artifacts directory (`HBX_ARTIFACTS`).

### Main model

- **What it is.** DeBERTa-v3-small (`microsoft/deberta-v3-small`, revision `a36c739020e01763fe789b4b85e2df55d6180012`) with a new output layer over the 28 jobs, fine-tuned on bios.
- **Training data.** 60,000 bios of the training subsample; 10,000 dev bios choose the weights.
- **Recipe.** AdamW, a linear warm-up over 6% of the steps and then a linear decay, bfloat16 autocast, 5 epochs, an effective batch of 32 (4 bios per step, 8 steps added up), at most 256 tokens per bio. Learning rate 3e-5, after also trying 5e-5; the two differ by less than the noise between neighbouring evaluations. Seed 42. The weights kept are those of the evaluation with the best dev macro-F1, at step 9,375.
- **Fit to the training bios.** Macro-F1 is 0.929 on the training bios, 0.788 on the dev bios and 0.783 on the test bios. I judge overfitting by three things. Dev and test macro-F1 agree, and I chose the model and the learning rate on dev. The gap between training and test macro-F1 is 0.145, about half the baseline model's 0.271 ([ADR 0002](decisions/0002-models.md)). Dev macro-F1 reached its best value at the last step, while dev loss rose slightly after epoch 3, from 0.59 to 0.64. I did not use it for early stopping. The model grew more confident, and its accuracy on bios it had not seen did not fall (`results/training_log.csv`).
- **Cost.** 6,167 s for the chosen run on the RTX 3050 laptop GPU (4 GB); 12,411 s for both learning rates.

### Baseline model

- **What it is.** Word counts (TF-IDF over single words and word pairs) with logistic regression, for comparison and because its weights can be read directly.
- **Recipe.** Words must appear in at least 5 bios, and the regularisation C is 256; both were chosen on the dev bios. It trained in 297 s on the CPU.

### Scrubbed model

- **What it is.** The main model's recipe, with one change: the pronouns and titles are removed from the training and dev bios (`data.scrub`). Learning rate 3e-5, 5 epochs, seed 42. The weights kept are those of the best dev macro-F1, 0.781 at step 8,250. It is saved as `deberta-scrubbed`. See [ADR 0007](decisions/0007-mitigation.md).

## Intended use

A research audit of screening bias: how a model that decides which job a bio describes treats women and men, and which words and internal states explain it. The models exist to be examined. The audit tests whether the explanations can be trusted, and what a first fix changes.

## Out-of-scope uses

- Screening, ranking or rejecting real people. The models were trained on 60,000 web bios, not on a hiring process, and they shortlist qualified women and men at different rates.
- Showing that a screener is fair. A small gap on this data does not carry over to another data set, another job or another model.
- Judging anyone's gender, origin or ability from a bio.
- Any decision about non-binary people: the data has two gender labels only.

## Factors

- **Gender, for all models.** The label comes from the pronouns of the bio, as in Bias in Bios ([data card](data_card.md)); it is binary. Women are the group whose gap is measured against men, and a positive gap means the model favours women.
- **Job.** The 28 jobs differ in the share of women. Six are studied in detail: surgeon, software engineer and composer (mostly men), dietitian, nurse and model (mostly women).
- **Origin, for the LLM only.** Origin cannot be read from the bios. It enters only through names added above a bio, from four origin groups (Spain and Hispanic America merged, because the model could not tell their names apart, Morocco, Romania and the United Kingdom).

## Metrics

- **Accuracy.** The share of bios for which the decision is the true job. For a recruiter: of 100 bios the model sorts, how many go to the right search.
- **Macro-F1.** The F1 score of each job averaged over the 28 jobs, so small jobs count as much as large ones. For a recruiter: whether the model also works for the rare jobs, not only for the common ones.
- **True positive rate (TPR).** Among the qualified candidates of a job, the share the model shortlists. For a recruiter: of 100 qualified surgeons, how many the search shows.
- **TPR gap and RMS TPR gap.** The TPR gap of a job is the TPR of women minus the TPR of men; above 0 the model favours women. The RMS TPR gap is the root mean square of the 28 jobs' gaps, one number for the overall shortlist gap whichever gender is favoured. For a recruiter: how far the search is from showing the same share of qualified women and qualified men; shuffling the genders at random gives a gap of about 0.04. The RMS TPR gap is a typical size of the per-job difference between the two shares: 0.148 means about 15 points on a typical job. The main model's gap is 0.148 on the full test set, and 0.143 on the test bios outside the showcase subset, which the mitigation uses.

Intervals are 95% bootstrap intervals over the test bios (1,000 resamples). They do not cover the variation between training runs.

## Evaluation data

- **Test split.** 99,069 bios of Bias in Bios that no model saw during training. All three screeners are scored on all of it.
- **Showcase subset.** 1,721 test bios from the six jobs above, up to 150 per job and gender. The explanations run on it.
- **Mitigation.** The scrubbed model and the proxy-word masking are also scored on the 97,348 test bios outside the showcase subset, because I chose the proxy words on the subset.
- Details, class balance and known issues are in the [data card](data_card.md).

## Results per group, with intervals

All of the test split. The shortlist gap is the RMS TPR gap; the last two rows score the scrubbed model on raw and on scrubbed bios.

| Model | Right job, of 100 bios | Macro-F1 | Shortlist gap (RMS TPR gap) |
|---|---|---|---|
| Baseline model | 79.5 [79.3, 79.8] | 0.728 [0.724, 0.733] | 0.167 [0.151, 0.190] |
| Main model | 83.7 [83.5, 83.9] | 0.783 [0.779, 0.788] | 0.148 [0.134, 0.172] |
| Scrubbed model, raw bios | 83.6 [83.4, 83.8] | 0.779 [0.775, 0.784] | 0.113 [0.102, 0.138] |
| Scrubbed model, scrubbed bios | 83.7 [83.4, 83.9] | 0.778 [0.774, 0.783] | 0.118 [0.104, 0.143] |

The main model, per job, for the six jobs studied in detail. "Shortlisted" is the TPR: of 100 qualified women and of 100 qualified men, how many the model shortlists. The gap is women minus men, in points.

| Job | Qualified women | Qualified men | Women shortlisted, of 100 | Men shortlisted, of 100 | Gap, in points |
|---|---|---|---|---|---|
| surgeon | 504 | 2,894 | 56.0 [51.8, 60.4] | 75.7 [74.2, 77.1] | -19.7 [-24.0, -14.9] |
| software engineer | 273 | 1,455 | 73.6 [67.9, 78.9] | 73.5 [71.1, 75.9] | +0.2 [-5.7, +6.0] |
| composer | 230 | 1,171 | 87.8 [83.6, 92.3] | 87.1 [85.1, 89.0] | +0.7 [-4.2, +5.6] |
| dietitian | 918 | 71 | 84.4 [81.8, 86.6] | 59.2 [47.8, 69.7] | +25.3 [+14.3, +37.1] |
| nurse | 4,304 | 434 | 84.1 [83.0, 85.2] | 73.3 [69.3, 77.1] | +10.8 [+6.8, +15.2] |
| fashion model | 1,550 | 324 | 86.1 [84.4, 87.9] | 42.6 [37.3, 48.2] | +43.5 [+37.7, +48.8] |

What this means: of 100 qualified women surgeons the model shortlists 56, and of 100 men 76, so a recruiter searching for surgeons sees fewer of the qualified women. For nurses it is the other way round, and in the job "fashion model" it shortlists 86 of 100 women and 43 of 100 men. Software engineer and composer show no gap that this data can tell from zero. The more a job is held by one gender, the more the model favours that gender (correlation 0.85, 95% interval 0.71 to 0.93 over the 28 jobs). The gaps of all 28 jobs are in `results/metrics_deberta.json`, and the other results are in [results.md](results.md).

## Ethical considerations

- **Gender is binary and inferred from pronouns.** The audit says nothing about non-binary people, and the label is partly the text itself.
- **The bios are real people's public web text.** I added no personal data and did not look for any. The names I add above bios come from public statistics lists (INE) and are not tied to any person.
- **A fairness measurement is not a certificate.** A job platform that ranks candidates with a model falls under the EU AI Act as a high-risk system (Annex III). This audit shows how a gap can be measured and explained; it does not show that a model is fit for that use.
- **A first fix is not a cure.** Removing the pronouns narrows the gap by about a fifth and leaves most of it ([results.md](results.md#f7-does-removing-the-cues-close-the-gap)). I would not present it as making the screener fair.
- **Gap and accuracy do not move together.** The more accurate model is not clearly the fairer one.
- **The LLM's stated reasons are not its causes.** The audit reads a reason as text, not as an account of what drove the decision.
- Real client data is not used anywhere in the project.

## Caveats and limitations

- **One training run per model.** A difference between two models includes run-to-run variation that I did not measure.
- **A subsample.** The models train on 60,000 training bios, a subsample of the training split, so their accuracy is below what the full split gives.
- **Bios are not CVs, and the task is a proxy.** They are short professional summaries from the web, in English, and the task is predicting an occupation. That is a proxy for screening, not a real hiring decision, and the findings carry over to CV screening only as far as that proxy holds.
- **Few bios in some groups.** The test split holds 71 men dietitians, so that job's gap has a wide interval and cannot be measured outside the showcase subset.
- **The explanations hold in aggregate.** Single Shapley values change with how they are computed; the conclusions rest on totals over hundreds of bios.
- **Cues beyond pronouns remain.** First names, topics and website names carry gender after scrubbing, and a probe still reads gender from the main model's internal states.
- **Labels are not audited.** Neither the job label nor the gender label was checked by a person.

## The LLM screener audited in F8

I audited this model from outside, as a black box. Full results are in [results.md](results.md#f8-does-an-llm-screener-react-to-names) and the design in [ADR 0008](decisions/0008-llm-audit.md).

- **Identity.** Ternary Bonsai 2 9B by Prism ML: a 2-bit ternary version of Qwen3.5-9B (`Qwen/Qwen3.5-9B`). File `Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf` from `prism-ml/Ternary-Bonsai-2-9B-gguf` on Hugging Face, Apache-2.0 licence. Served on the laptop by llama-swap with a llama.cpp fork from Prism ML, under the name `pc-equilibrado`.
- **Settings.** Temperature 0, seed 42, thinking off, presence penalty 0, at most 400 tokens. For a job ad and a bio it answers `{"reason": ..., "decision": "yes" or "no"}`. The prompts and job ads are in `prompts/`.
- **What I measured.** 600 profiles, 100 for each of the six jobs: 70 qualified candidates and 30 who are not. Each goes through eleven presentations (arms): the bio alone, a neutral ID line, the pronouns swapped, and a name from each of four origin groups, for each gender. Of 100 qualified candidates it says yes to 59 with no header (interval 55 to 64). The primary contrast, a woman's name minus a man's name, is -0.5 points (interval -4.7 to +3.5), so this data shows no gap of 5 points or more, and the interval is wide. The contrast pools the six jobs, so effects of opposite sign in different jobs could cancel. Any name at all lowers the yes-rate by 6.0 points against no name (interval 2.1 to 10.1), a lead I did not plan for, and a confounded one: some bios already contain a name, for example a profile that opens with "Mrs." and a full name, and adding a name sometimes changes the pronouns. Rewording the instruction moves the yes-rate to 47 and 39 of 100.
- **Cost.** 8,167 calls in 2.99 hours on the laptop.
- **Limits.** One model, one prompt, six jobs, 180 profiles that are not qualified. A gap of one or two points could hide in these intervals.
