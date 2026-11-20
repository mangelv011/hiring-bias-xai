# 0006. Probing for gender and activation patching

Status: accepted. Implemented in `src/hiring_bias_xai/probing.py` and `patching.py` (statistics), `models.py` (`DebertaModel.batches`, `representations`, `states`, `patched_logits`); run by `scripts/probing_features.py`, `scripts/probing.py` and `scripts/patching.py`; drawn by `scripts/figure_f6.py`.

## Context

Two questions about the inside of the main model. Is gender still readable from its representations when the pronouns are gone? And does what the model holds about gender change its decision? A probe answers the first. It is descriptive: a probe that reads gender shows the information is there, not that the model uses it (Belinkov 2022). Activation patching answers the second. It is causal: it puts the states of one run into another and measures what changes (Meng et al. 2022; Heimersheim and Nanda 2024).

## Decision

### Probing

**Representations.** For each bio, the state after the embeddings and after each of the six transformer layers (seven vectors of 768), each averaged over the non-padding tokens, [CLS] and [SEP] included. The mean is used for both encoders because the [CLS] state of the pretrained model was never trained to summarise anything. Features are stored as float16 in `HBX_ARTIFACTS`; the forward pass runs in bfloat16 on the GPU, like every other score in the project.

**Text.** `data.scrub`: gendered pronouns and titles removed, as in the removal control of ADR 0005. The probe is fitted on 10,000 scrubbed train bios, a sample of the 60,000-bio training subsample stratified by occupation and gender in proportion to the group sizes (`data.stratified_sample`, seed 42; ids in `results/probing_train_ids.csv`). It is scored on the scrubbed showcase subset (1,721 bios). Two encoders: `microsoft/deberta-v3-small` at the pinned revision with no fine-tuning (same tokenizer and maximum length of 256), and the main model.

**Probe.** Logistic regression on standardised features. The regularisation C comes from the grid 1e-4, 1e-3, 1e-2, 1e-1, 1 by 5-fold cross-validated accuracy on the training bios; the scaler is fitted inside each fold, and a tie goes to the smaller C. A linear probe keeps the claim modest: what a linear map reads is information a later layer can read too, and a flexible probe can learn the task itself (Hewitt and Liang 2019).

**Test accuracy and interval.** Accuracy on the showcase subset, with the 95% percentile interval of 1,000 bootstrap resamples of the bios (seed 42).

**Permutation test.** The training genders are shuffled within each occupation, the probe is refitted and scored on the real test genders, 100 times per layer and model. C is not chosen again for each shuffle; it is the one chosen on the real labels. The p-value is (1 + shuffles at least as accurate as the probe) / (1 + 100), so the smallest value is 1/101 = 0.0099. A shuffle within occupation keeps what the occupation says about gender in the training data and removes the rest. The null probe can therefore learn "this is a nurse bio, say female", and its accuracy sits near the occupation-only reference. A small p-value says the probe reads gender beyond what the occupation tells. It is a stricter test than shuffling over all bios, where any probe that uses the topic of the bio would pass.

**References**, on the same split:

- *Majority class.* The most common training gender. It is the floor.
- *Occupation only.* The majority gender of the true occupation in the training bios. The showcase occupations were picked at the extremes of the share of women, so this reference is high, and it is the honest bar: a probe below it has not read more than the topic.
- *TF-IDF.* Word 1-2 grams with logistic regression on the same scrubbed text, C chosen by the same 5-fold rule on its own grid (0.1, 1, 10, 100). It shows what the words give without any network. A probe at this level reads the words, as a bag of words does.

The references have no p-value.

**Two subsets.** Every number is reported for all showcase bios and for those without a first name, using the first-name rule of ADR 0005 (`counterfactual.first_names_in`, on the original text). A name tells gender and scrubbing leaves it in. The probe is fitted once; only the scored bios change.

### Activation patching

**Pairs.** Clean is the original showcase bio, corrupted is `swap_gender` of it (ADR 0005). Only the main model is patched. A pair is kept when the two token sequences have the same length after the cut to the maximum, so that positions line up. The number of pairs dropped is reported, for all bios and for the eligible ones.

**Eligible bios.** A bio is eligible when the absolute change of its true-class log-odds under the swap is above the 95th percentile of the absolute change under the removal control over all 1,721 bios. Both changes are recomputed from the model with the definitions of ADR 0005. The removal control moves the log-odds by editing the text, not by changing gender, so its spread is the noise a swap effect has to clear. The number of eligible bios is reported.

**Patch.** For each layer output (the embeddings, then each transformer layer), the clean run's hidden states replace the corrupted run's, (a) at the token positions where the two sequences differ, (b) at all positions. This is denoising: clean into corrupted. It asks whether the states at these positions are sufficient to restore the clean behaviour. It does not say they are necessary: another route could carry the effect if these states were removed (Heimersheim and Nanda 2024). The reverse direction, noising, is not run.

**Restored share**, per bio: (patched − corrupted) / (clean − corrupted) of the true-class log-odds. Mean and median over the eligible bios, each with a 95% percentile interval of 1,000 bootstrap resamples of the bios. A share can leave [0, 1]; the median is reported because the ratio has heavy tails when the effect is small.

**Three results hold by construction.** They check the hook; they are not findings.

1. At the embeddings, a position's state depends only on its token. Patching all positions, or only the differing ones, gives 100%.
2. Patching all positions at any layer replaces everything the later layers read, so the clean run is reproduced and the share is 100% at every layer. Panel (b) of F6 is therefore a flat line at 100%; it confirms the hook layer by layer.
3. The classifier reads the [CLS] state of the last layer, and [CLS] is never a differing position. At the last layer the differing positions give 0%.

The informative result is the share at the differing positions between the embeddings and the last layer: how fast the effect of the swap moves to positions that were not edited.

## Expected outcomes

Written before the probes and the patching ran. Whatever comes out is reported, and nothing is tuned afterwards. One timing run of the probe on a single layer of the main model (layer 4, 16 shuffles, to measure the cost) was seen first: its accuracy was 0.607 against a null mean of 0.560. The expectations below do not use it.

1. Gender is readable above the occupation-only reference from scrubbed text: in at least one layer of each encoder the probe's interval lies above the reference accuracy, with a permutation p-value at the minimum, 1/101.
2. The embedding layer (a bag of word vectors) already reads gender at about the TF-IDF level, because the words carry it. Later layers of the pretrained model do not read much more than that.
3. The main model's last layers read gender less than the pretrained model's middle layers, since fine-tuning on occupations pushes the final states toward occupation.
4. Without the bios that carry a first name, accuracy is a few points lower and still above the occupation-only reference.
5. Patching the differing positions restores most of the effect in the early layers (above 50% at the embeddings and the first two layers) and less in later ones, falling to 0% at the last layer, as the information spreads to positions that were not edited.
6. The all-positions line is 100% at every layer.

## Alternatives considered

- **LEACE and INLP.** Methods that remove a concept from the representation: LEACE (Belrose et al. 2023) in closed form, INLP (Ravfogel et al. 2020) by repeated projections. They answer a different question, what happens to the decision when gender is erased. Not run; they belong to the mitigation work.
- **Control task instead of a permutation test.** A probe on random labels (Hewitt and Liang 2019) measures how much the probe can memorise. The permutation within occupation tests the claim we make, gender beyond occupation, and costs the same.
- **A nonlinear probe.** Higher accuracy, but it can compute the answer itself. Not run.
- **[CLS] pooling.** The pretrained model's [CLS] state is untrained. Not run.
- **Choosing C again for each shuffle.** 100 times the cost for a small change in the null. C is fixed at the value chosen on the real labels.
- **Noising direction, path patching, single heads.** They would say where the effect travels. This issue asks whether the representation changes the decision.

## Consequences

- A probe above the references shows that gender can be read from the layer. It does not show that the model uses it. Patching is the causal test, and only for the swap edit and the differing tokens.
- The probe is trained on 28 occupations and scored on six that sit at the extremes of the share of women, so the occupation-only reference is high. The result for other occupations is not tested.
- The permutation p-value cannot go below 1/101 with 100 shuffles. A layer at that value has beaten every shuffle, which a larger count could separate further.
- The eligible bios are those the swap moves most. The restored share is a statement about them, not about all bios.
- Pairs of different token length are dropped, so the patched bios are those whose swap did not change the tokenisation. A pronoun that splits into more tokens in one gender would be missing.
- The features and the patched runs use bfloat16 on the GPU. The embedding check (100% at every layer, all positions) shows the numerical noise is small against the effect.
- The scripts read only files this project wrote (the features as `.npy` arrays without pickle, the saved model) and write to `results/`; no untrusted input is involved.

## References

- Belinkov, Y. (2022). Probing Classifiers: Promises, Shortcomings, and Advances. Computational Linguistics 48(1).
- Hewitt, J. and Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. EMNLP.
- Meng, K. et al. (2022). Locating and Editing Factual Associations in GPT. NeurIPS.
- Heimersheim, S. and Nanda, N. (2024). How to use and interpret activation patching. arXiv:2404.15255.
- Belrose, N. et al. (2023). LEACE: Perfect linear concept erasure in closed form. NeurIPS.
- Ravfogel, S. et al. (2020). Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection. ACL.
