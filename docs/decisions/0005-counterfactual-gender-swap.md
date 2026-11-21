# 0005. Counterfactual gender swap, residual cues and the removal control

Status: accepted. Implemented in `src/hiring_bias_xai/counterfactual.py` (swap and cues), `counterfactual_effects.py` (metrics), `first_names.py` (name list); run by `scripts/counterfactual_gender.py`, drawn by `scripts/figure_f4.py`.

## Context

Changing only the gender of a candidate and checking whether the decision changes is the most direct test of discrimination (Kusner et al. 2017; Wachter et al. 2017). The bios carry pronouns, not names, so the edit is a pronoun swap. A swap changes several tokens at once, so a flip does not prove the model uses gender: the edit itself can move the score. Two controls answer that. The removal control deletes the same words instead of swapping them. The residual-cue flags mark the bios where the swap leaves gender information in the text.

## Decision

**Swap.** `swap_gender(text)` swaps he and she, him and her, himself and herself, hers and his, and the titles Mr, Ms and Mrs (Mrs gives Mr). Contractions keep their suffix (he's gives she's). Capitalisation follows the original word: lower case, Capitalised or ALL CAPS. Matching is on whole words, so there, sheriff, theme, history and Herb stay. Titles are swapped only when written Capitalised, because MS is a degree and ms a unit.

"her" is a possessive determiner (his) or an object pronoun (him), and "his" is a determiner (her) or a pronoun (hers, "the book is his"). The tagger reads "her". For "his" the tag is unreliable (spaCy gave PRP and PRP$ for the same use), so a "his" is a pronoun when it ends the text or the next word is punctuation, a conjunction, a preposition, an auxiliary or a particle. A "her" or "his" that the tagger does not read as a word of its own (inside a web address such as `praxis-his.com`) stays.

**Tagger.** spaCy 3.8 with `en_core_web_sm` 3.8.0, tagger only. Both are MIT licensed. The model is a wheel pinned by URL and sha256 in `pyproject.toml` (a PEP 508 direct reference), so `pip install -e .` and `uv sync` install it with no download step at runtime or in CI. Swapping the 1,721 showcase bios took 6.8 s on the laptop CPU, one text at a time.

**Residual cues**, fixed in code (`RESIDUAL_NOUNS`) before any swapped bio was read: gendered nouns and titles, whole words, any case. They are not swapped; a bio that contains one is flagged.

- Female: mother(s), mom(s), mum, mama, wife, wives, actress(es), businesswoman/women, daughter(s), sister(s), girl(s), girlfriend, woman, women, lady, ladies, queen(s), chairwoman/women, spokeswoman, congresswoman, policewoman, saleswoman, waitress, stewardess, hostess, heroine, grandmother, grandma, aunt, niece, bride, widow, madam, madame, princess, female(s), matriarch, gal.
- Male: father(s), dad(s), husband(s), actor(s), businessman/men, son(s), brother(s), boy(s), boyfriend, man, men, gentleman/men, king(s), chairman/men, spokesman, congressman, policeman, salesman, waiter, steward, grandfather, grandpa, uncle, nephew, groom, widower, sir, prince, male(s), patriarch, guy(s).
- A first name: the first word of a span that spaCy's entity recogniser labels PERSON, when that word is in the SSA list (names with at least 10,000 births in 1880 to 2025, 2,808 names; data card). The first rule used the tagger's proper-noun tag and flagged 1,072 of the 1,721 bios; place names on the list (America 24 bios, Austin 8, Asia 7) were among the commonest cues. That count was seen before reading any swapped bio. The rule was then changed to the entity recogniser, and the 225 noun and 631 name flags in `results/counterfactual_gender.json` come from the final rule. The recogniser misses some lone first names (it labelled "Maria" as an organisation and a place in short test sentences), so the name flag is a lower bound. A surname that is also a first name counts.

**Removal control.** `data.scrub`, the De-Arteaga et al. (2019) list: the same bios with the gendered pronouns and titles deleted.

**Metrics**, per screener (main and baseline model), per showcase occupation and pooled, on all 1,721 showcase bios and on the unflagged ones, with 95% percentile bootstrap intervals over bios (1,000 resamples, `config.SEED`):

- flip rate: share of bios whose predicted occupation changes;
- mean change of the log-odds of the true occupation (`class_log_odds`), edited minus original;
- flip direction, swap only: among flipped bios, the share whose new prediction has a training share of women (`data.share_women`, training subsample) that moves toward the new gender. A woman's bio turned male is toward when the share falls. This is the share F1 uses, so the direction result and F1 are not independent. The six showcase occupations sit at the extremes of that share, so many flips move it in one direction whatever the edit; the removal flips get the same measure, read as if the new gender were the opposite one, as the reference.

Bios the swap does not change (no pronoun) stay in the denominators; 1,689 of 1,721 are changed by the swap and 1,706 by the removal.

**Hand check.** 50 bios drawn at random among the bios the swap changes (seed `config.SEED`), read as original and swapped sentences, each judged coherent (grammatical, same meaning apart from gender) or not, with a reason for every incoherent one: `results/counterfactual_hand_check.csv`. The share judged incoherent gets a Wilson interval. A residual cue in a coherent bio is a note, not an incoherence: it is what the flags are for.

## Alternatives considered

- **NLTK averaged perceptron tagger.** Apache 2.0 and lighter, but its data comes from `nltk.download` at run time with no hash, so CI needs a download step and a pin of its own. Not measured.
- **No tagger: "her" is "his" before a word and "him" otherwise.** No dependency, but "her" before a noun at the end of a clause and "gave her books" are wrong. Not measured.
- **Swapping the residual nouns too** (mother for father). The pairs are not one to one (actor, hostess, king) and it would make the hand check a test of my dictionary. They are flagged instead.
- **Dropping flagged bios from the main result.** Names are in 631 of 1,721 bios, so the unflagged set is smaller and different. Both sets are reported.
- **SSA names from `ssa.gov` directly.** The server returns HTTP 403 to scripts. The pin is an Internet Archive capture of the same URL, checked by sha256.

## Consequences

- A flip under the swap is not evidence of gender use by itself. Read it against the removal control: where the two rates match, the edit explains the flips.
- Failure types seen in the hand check: a "his" before an opening quotation mark turns into "hers"; an object pronoun that refers to someone else ("making him well-informed", about "the total man") is swapped with the rest; a swapped pronoun can become ambiguous with another person named in the text. The hand check was done once, on the delivered `swap_gender`; any later change to the swap needs a new check.
- Names stay in the text after the swap (Rosalind with "his"), as do words outside the list (hunk). The unflagged subset is a subset with no detected cue, not with no cue.
- The swap is a deterministic function of the bio, so the effects analysis rebuilds the pairs; no swapped text is stored in the artifacts.
- The name-list download is a network boundary. The lower-trust input is the archive; the pinned sha256 is checked before anything is written or read, only `https` URLs are fetched, the download and the unpacked size are capped, only members named `yobYYYY.txt` are read, as text in memory (nothing is extracted or executed), and a line that is not `name,sex,count` stops the parse. Tests cover a hash mismatch, a non-https URL, an oversized archive and malformed lines.
- Name swaps are out of scope; they belong to the LLM audit ([ADR 0008](0008-llm-audit.md)).

## Outcomes

The swap changed 1,689 of the 1,721 bios and the removal control 1,706. For the main model the swap flips 6.9% of decisions and the deletion 6.3%, with overlapping intervals in every job, so this data shows no reaction to the gender itself. The baseline model flips 13.5% under the swap and 8.8% under the deletion. 752 bios carry a residual cue (225 by a noun, 631 by a first name); leaving them out changes no conclusion. Of the 50 swapped bios I read, 3 were incoherent, for the reasons listed above. Flips after a swap lean towards the new gender more than flips after a deletion do (88% against 72% for the main model). The swap was reused for the patching pairs (ADR 0006) and for the pronoun arm of the LLM audit (ADR 0008).
