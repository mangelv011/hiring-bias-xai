# Data card: Bias in Bios

Follows the questions of Gebru et al., "Datasheets for Datasets" (2021), cut to what this project needs. Terms follow [GLOSSARY.md](../GLOSSARY.md).

## Source

- Dataset: Bias in Bios (De-Arteaga et al., 2019), in the version of Ravfogel et al. (2020) published as `LabHC/bias_in_bios` on Hugging Face.
- Pinned revision: `052f01de644dba841176e0449528b41f27d94a61`. To move to another revision, change the pin in `src/hiring_bias_xai/data.py` and rerun `scripts/prepare_data.py`.
- Licence: MIT (dataset card). The Ravfogel version drops 5,557 bios of the original release that were no longer available.
- Cached outside the repository, in `HBX_ARTIFACTS` (default `~/.cache/hiring-bias-xai`). No raw data is tracked by git.
- Columns: `hard_text` (the bio without its title sentence), `profession` (0 to 27), `gender` (0 or 1). The module renames them `text`, `occupation` and `gender`, and maps the integers to names with the coding below.

## Size

| Split | Official size | Used here |
|---|---|---|
| train | 257,478 | 60,000 (training subsample) |
| dev | 39,642 | 10,000 (training subsample) |
| test | 99,069 | 99,069 (full) |
| XAI subset | | 2,000 bios drawn from test |

Subsamples are stratified by occupation x gender with proportional allocation and the global seed (42). Group sizes are rounded by largest remainder, so the totals are exact. Bio IDs are `<split>-<row index in the official split>`, for example `test-1234`; the XAI subset IDs are in `results/xai_subset_ids.csv`.

## How the labels were produced

- **Occupation**: taken by the dataset authors from the title sentence of the bio (for example "Nancy Lee is a registered nurse."). That sentence is removed from `text`, so the model has to infer the occupation from the rest. 28 occupations, coded alphabetically from `accountant` = 0 to `yoga_teacher` = 27 (the table in the dataset card, copied into `data.py`).
- **Gender**: inferred by the authors from the pronouns of the bio. Binary only. Coding `male` = 0, `female` = 1, checked on the train split: 92.3% of the female bios contain "she" or "her" and none contains "he" or "his"; 96.7% of the male bios contain "he" or "his" and none contains "she" or "her". Nurse bios are 90.8% female in the training subsample, software engineer bios 15.8%, as expected from the occupation coding.
- Neither label was checked by a person for this project.

## Class balance

Full counts per split, occupation and gender, and the share of women per occupation in the training subsample, are in `results/data_summary.json`.

- Women are 46.1% in train, dev and test (27,661 of 60,000 in the training subsample; 923 of 2,000 in the XAI subset).
- Occupations are skewed. In the test split `professor` has 29,520 bios and `physician` 10,250, while `rapper` has 351, `personal_trainer` 358, `interior_designer` 366 and `dj` 372.
- The share of women ranges from 9.4% (`rapper`) to 92.8% (`dietitian`) in the training subsample.
- Showcase occupations (rule in `results/showcase_occupations.json`): lowest share of women `surgeon` (14.8%), `software_engineer` (15.8%), `composer` (16.4%); highest `dietitian` (92.8%), `nurse` (90.8%), `model` (82.7%). `rapper` and `dj` are more male-dominated but have fewer than 500 test bios, so they are not eligible.

## Scrubbing

`scrub` removes `he, she, her, his, him, hers, himself, herself, mr, mrs, ms` as whole words, ignoring case, with attached `'s`, `'d`, `'ll` or a title's period. This is the list of De-Arteaga et al. (2019), section 4. Their first-name removal is not applied: the bios have no name field and the loader does not detect names inside the text. `Ms` also removes the degree abbreviation "MS" (Master of Science), as in the original list.

## Known issues

- **Binary gender.** The audit says nothing about non-binary people.
- **The gender label is derived from the text.** The pronouns that define the label are in the input, so a model can read it directly. The scrubbed variant and the probing control task address this; scrubbing leaves other signals (names inside the text, topics, wording).
- **First names stay in some bios.** Example seen while checking the data: "Brittni graduated ... with her Bachelor's Degree". This was not counted. It leaks gender and sometimes origin, and the audit does not use it as an origin signal; origin groups enter only through counterfactual name headers.
- **Source is web text.** English bios from Common Crawl, written by or for people in these occupations. They are short professional summaries, not CVs.
- **Skewed occupations**, as above. Rare occupations have few test bios, so their per-occupation metrics are noisy.
- **Label noise.** The occupation label comes from the title sentence and was not audited here.
- **Training subsample.** 60,000 of 257,478 train bios keep the occupation x gender proportions but drop information, so accuracy is below what the full set gives.
