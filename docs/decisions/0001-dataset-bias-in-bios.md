# 0001. Dataset: Bias in Bios

Status: accepted

## Context

The audit needs texts that look like CVs, a ground-truth occupation per text, and a gender label that does not come from the model under test. Pronouns must be present in the text so that counterfactual swaps and scrubbing have something to act on.

## Decision

Use Bias in Bios (De-Arteaga et al., 2019) through the Hugging Face dataset `LabHC/bias_in_bios` (MIT): about 396k English biographies, 28 occupations, binary gender inferred from the pronouns in each bio. The task is framed as a screening decision: a recruiter searches for an occupation and the model decides which bios match.

## Alternatives considered

- **Synthetic resumes or LLM-labelled hiring data.** Circular: the labels and the bias would come from the generator, so there is no ground truth to audit against.
- **Real hiring outcome data.** The only data that measures actual hiring decisions, but it is not public.
- **JobFair and other fairness benchmarks.** Smaller, built for LLM evaluation, and without a large labelled training set for a fine-tuned classifier and for gradient-based explanations.

## Consequences

- Gender is binary and inferred from pronouns, so the audit cannot say anything about non-binary people, and the label is partly the text itself. Scrubbing and the probing control task exist for this reason.
- Origin cannot be read from the bios. Origin groups enter only through counterfactual name headers built from Spanish INE name lists.
- Bios are short professional summaries, not full CVs. Findings transfer to CV screening only as far as that gap allows, and the report states the limit.
- The occupation shares of women in the data are skewed, which the TPR gap analysis uses on purpose.
