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

- Gender is binary and inferred from pronouns, so the audit cannot say anything about non-binary people, and the label is partly the text itself. Scrubbing and the permutation test of the probes exist for this reason.
- Origin cannot be read from the bios. Origin groups enter only through counterfactual name headers built from Spanish INE name lists.
- Bios are short professional summaries, not full CVs. Findings transfer to CV screening only as far as that gap allows, and the report states the limit.
- The occupation shares of women in the data are skewed, which the TPR gap analysis uses on purpose.

## Outcomes

The dataset did what this record asked of it. At the pinned revision it gave 257,478 train, 39,642 dev and 99,069 test bios; the Ravfogel version leaves out 5,557 bios of the original release. Gender code 1 is female, checked on all three splits: in train, 92.4% of the code-1 bios contain a feminine pronoun and none a masculine one, and 97.1% of the code-0 bios contain a masculine pronoun and none a feminine one. The share of women per occupation runs from 9.4% (rapper) to 92.8% (dietitian), which is the spread the TPR gap analysis uses. The limits named above held: 631 of the 1,721 showcase bios still contain a first name, and the audit says nothing about non-binary people. Scrubbing and the permutation test of the probes (ADR 0006) are the two answers to the gender label being part of the text. The control task I first named for the probes was not run. Details are in the [data card](../data_card.md).
