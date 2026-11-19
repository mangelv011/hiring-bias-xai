# Data card: Bias in Bios

Follows the questions of Gebru et al., "Datasheets for Datasets" (2021), cut to what this project needs. Terms follow [GLOSSARY.md](../GLOSSARY.md). Every number below comes from `results/data_summary.json` and `results/showcase_occupations.json`, produced by `scripts/prepare_data.py`, unless a sentence says otherwise.

## Source

- Dataset: Bias in Bios (De-Arteaga et al., 2019), in the version of Ravfogel et al. (2020) published as `LabHC/bias_in_bios` on Hugging Face.
- Pinned revision: `052f01de644dba841176e0449528b41f27d94a61`. To move to another revision, change the pin in `src/hiring_bias_xai/data.py` and rerun `scripts/prepare_data.py`.
- Licence: MIT (dataset card). The Ravfogel version leaves out 5,557 bios of the original release that were no longer available.
- The cache lives outside the repository, in `HBX_ARTIFACTS` (default `~/.cache/hiring-bias-xai`). Git tracks no raw data.
- Columns: `hard_text` (the bio), `profession` (0 to 27), `gender` (0 or 1). The module renames them `text`, `occupation` and `gender` and maps the integers to names with the coding below.
- Collection, per De-Arteaga et al. and not re-checked here: English bios from Common Crawl.

## Size

| Split | Official size | Used here |
|---|---|---|
| train | 257,478 | 60,000 (training subsample) |
| dev | 39,642 | 10,000 (training subsample) |
| test | 99,069 | 99,069 (full) |
| XAI subset | | 2,000 bios drawn from test |

Subsamples are stratified by occupation x gender with proportional allocation and the global seed (42). Group quotas are rounded down and the leftover bios go to the groups with the largest remainders, so the totals are exact. Groups are visited in sorted (occupation, gender) order, so `female` comes before `male`. Bio IDs are `<split>-<row index in the official split>`, for example `test-1234`; the XAI subset IDs are in `results/xai_subset_ids.csv`.

In the dev split a bio has 150 to 961 characters (median 358).

## How the labels were produced

- **Occupation**: the dataset authors took it from the first sentence of the bio, which states the title ("<name> is a <title>"), per De-Arteaga et al. That sentence is not in `text`: in the dev split the most common first words are "He" (9,968 bios), "She" (9,163), "His" and "Her", so a model has to infer the occupation from the rest. 28 occupations, coded in alphabetical order from `accountant` = 0 to `yoga_teacher` = 27 (the table in the dataset card, copied into `data.py`).
- **Gender**: the authors inferred it from the pronouns of the bio. Binary only. Coding `male` = 0, `female` = 1, matching the dataset card and checked on all three splits. In train, 92.4% of the code-1 bios contain "she", "her", "hers" or "herself" and none contains "he", "his", "him" or "himself"; 97.1% of the code-0 bios contain a masculine pronoun and none contains a feminine one. Dev and test give the same figures within 0.1 points. The check is the slow test `test_real_gender_code_1_is_female_and_0_is_male`.
- No person checked either label for this project.

## Class balance

Full counts per split, occupation and gender, and the share of women per occupation in the training subsample, are in `results/data_summary.json`.

- Women are 46.1% in the training subsample (27,661 of 60,000), in dev (4,610 of 10,000) and in test (45,672 of 99,069), and 46.2% in the XAI subset (923 of 2,000).
- Occupations are skewed. In the test split `professor` has 29,520 bios and `physician` 10,250, while `rapper` has 351, `personal_trainer` 358, `interior_designer` 366 and `dj` 372.
- The share of women in the training subsample ranges from 9.4% (`rapper`) to 92.8% (`dietitian`).
- Showcase occupations (rule in `results/showcase_occupations.json`): lowest share of women `surgeon` (14.8%), `software_engineer` (15.8%), `composer` (16.4%); highest `dietitian` (92.8%), `nurse` (90.8%), `model` (82.7%). `rapper` (9.4%) and `dj` (14.2%) are more male-dominated but have fewer than 500 test bios, so they are not eligible.

## Scrubbing

`scrub` removes `he, she, her, his, him, hers, himself, herself, mr, mrs, ms` as whole words, ignoring case, together with an attached `'s`, `'d`, `'ll` or bare apostrophe on a pronoun and the period after a title. This is the list of De-Arteaga et al. (2019), section 4. Their first-name removal is not applied: the loader does not detect names inside the text. The title "Ms" also matches the degree abbreviation "MS" (Master of Science); 138 of the 39,642 dev bios contain the capitalised form.

## Known issues

- **Binary gender.** The audit says nothing about non-binary people.
- **The gender label is derived from the text.** The pronouns that define the label are in the input, so a model can read it directly. The scrubbed variant and the probing control task address this; scrubbing leaves other signals (names inside the text, topics, wording).
- **First names can stay in the text.** They leak gender and sometimes origin. This was not counted. The audit does not use them as an origin signal; origin groups enter only through counterfactual name headers.
- **Source is web text.** The bios are short professional summaries, not CVs.
- **Skewed occupations**, as above. Rare occupations have few test bios, so their per-occupation metrics are noisy.
- **Label noise.** The occupation label comes from the title sentence. This project did not audit it.
- **Training subsample.** 60,000 of 257,478 train bios keep the occupation x gender proportions but drop information, so accuracy is below what the full set gives.
