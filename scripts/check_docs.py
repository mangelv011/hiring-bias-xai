"""Check the written documents against the repository: `python scripts/check_docs.py`.

Reads files of this repository only and makes no network calls. Exits with 1 and prints one line
per problem when a link, an image path or a heading anchor does not resolve.
"""

import csv
import json
import os
import re
import sys
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`[^`\n]*`")
MARKDOWN_LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_REFERENCE = re.compile(r"""\b(?:src|href)\s*=\s*["']([^"']+)["']""")
HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
HEADING_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
HTML_TAG = re.compile(r"<[^>]*>")
NOT_IN_SLUG = re.compile(r"[^\w\s-]")
NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
EXTERNAL = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)


def documents(root: Path) -> list[Path]:
    """README.md, the pages of docs/ and the decision records, in a fixed order."""
    found = [root / "README.md"]
    found += sorted((root / "docs").glob("*.md"))
    found += sorted((root / "docs" / "decisions").glob("*.md"))
    return [path for path in found if path.is_file()]


def unfenced_lines(text: str):
    """Yield (line number, line) for the lines outside fenced code."""
    fenced = False
    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            yield number, line


def prose_lines(text: str):
    """Yield (line number, line) for the lines outside fenced code, inline code removed."""
    for number, line in unfenced_lines(text):
        yield number, INLINE_CODE.sub("", line)


def references(text: str):
    """Yield (line number, target) for every link and image of a document."""
    for number, line in prose_lines(text):
        for pattern in (MARKDOWN_LINK, HTML_REFERENCE):
            for match in pattern.finditer(line):
                yield number, match.group(1)


def slug(heading: str) -> str:
    """GitHub anchor of a heading: no markup or punctuation, lower case, spaces as hyphens."""
    text = HTML_TAG.sub("", HEADING_LINK.sub(r"\1", heading))
    return NOT_IN_SLUG.sub("", text.lower()).replace(" ", "-")


def anchors(text: str) -> set[str]:
    """The anchors of the headings of a document; a repeated heading gets -1, -2 and so on."""
    seen: dict[str, int] = {}
    found = set()
    for _, line in unfenced_lines(text):
        match = HEADING.match(line)
        if match:
            base = slug(match.group(1))
            count = seen.get(base, 0)
            seen[base] = count + 1
            found.add(base if count == 0 else f"{base}-{count}")
    return found


def exists_with_case(root: Path, relative: Path) -> bool:
    """True when every part of `relative` exists under `root` with exactly this spelling."""
    here = root
    for part in relative.parts:
        if not here.is_dir() or part not in {entry.name for entry in here.iterdir()}:
            return False
        here = here / part
    return True


def resolve(root: Path, document: Path, target: str) -> Path | None:
    """The path a relative target points to, or None when it is missing."""
    path = unquote(target.split("#", 1)[0])
    if not path:
        return document
    base = Path(os.path.normpath(root))
    here = Path(os.path.normpath(document.parent / path))
    try:
        relative = here.relative_to(base)
    except ValueError:
        return None
    return here if exists_with_case(base, relative) else None


def check_links(root: Path) -> list[str]:
    """One message per link or image of the documents whose target does not resolve."""
    errors = []
    for document in documents(root):
        name = document.relative_to(root).as_posix()
        text = document.read_text(encoding="utf-8")
        for number, target in references(text):
            if EXTERNAL.match(target):
                continue
            found = resolve(root, document, target)
            if found is None:
                errors.append(f"{name}:{number}: {target} does not resolve")
                continue
            _, _, fragment = target.partition("#")
            if fragment and found.suffix == ".md":
                if unquote(fragment) not in anchors(found.read_text(encoding="utf-8")):
                    errors.append(f"{name}:{number}: {target} has no such heading")
    return errors


class Cell(NamedTuple):
    """One number a document states and where it comes from in results/.

    `pattern` has one group holding the number as written. `path` is dotted keys for a JSON file
    (a list is entered by position) or `column=value,...:column` for a CSV file.
    """

    document: str
    pattern: str
    file: str
    path: str
    fmt: str


class Allowed(NamedTuple):
    """A phrase whose numbers are not results: a year, a count in a method description."""

    document: str | None
    pattern: str
    reason: str


NUMBER_DOCUMENTS = ("README.md", "docs/model_card.md")


def _fixed(scale: Decimal, decimals: int, signed: bool = False, absolute: bool = False):
    """A formatter: the value times `scale`, rounded half up to `decimals` places."""
    step = Decimal(1).scaleb(-decimals)

    def fmt(value) -> str:
        rounded = (Decimal(repr(value)) * scale).quantize(step, rounding=ROUND_HALF_UP)
        if absolute:
            rounded = abs(rounded)
        return f"{rounded:+f}" if signed else f"{rounded:f}"

    return fmt


def _count(value) -> str:
    return f"{int(Decimal(repr(value)).quantize(Decimal(1), rounding=ROUND_HALF_UP)):,}"


FORMATS = {
    "per100": _fixed(Decimal(100), 0),
    "abs_per100": _fixed(Decimal(100), 0, absolute=True),
    "pct1": _fixed(Decimal(100), 1),
    "abs_points1": _fixed(Decimal(100), 1, absolute=True),
    "signed_points1": _fixed(Decimal(100), 1, signed=True),
    "dec2": _fixed(Decimal(1), 2),
    "dec3": _fixed(Decimal(1), 3),
    "of150": _fixed(Decimal(150), 0),
    "seconds0": _fixed(Decimal(1), 0),
    "minutes0": _fixed(Decimal(1) / Decimal(60), 0),
    "minutes1": _fixed(Decimal(1) / Decimal(60), 1),
    "hours2": _fixed(Decimal(1) / Decimal(3600), 2),
    "count": _count,
    "learning_rate": lambda value: f"{value:.0e}".replace("e-0", "e-"),
}


def read_value(results_dir: Path, file: str, path: str):
    """The value at `path` of a results file; raises when the file, path or row is missing."""
    full = results_dir / file
    if full.suffix == ".csv":
        keys, _, column = path.rpartition(":")
        wanted = dict(item.split("=", 1) for item in keys.split(","))
        with full.open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if all(r[k] == v for k, v in wanted.items())]
        if len(rows) != 1:
            raise ValueError(f"{len(rows)} rows match")
        return float(rows[0][column])
    value = json.loads(full.read_text(encoding="utf-8"))
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _blank(line: str, pattern: re.Pattern) -> str:
    """The line with the first group of every match of `pattern` replaced by spaces."""
    for match in pattern.finditer(line):
        start, end = match.span(1)
        line = line[:start] + " " * (end - start) + line[end:]
    return line


def claim_lines(text: str) -> list[tuple[int, str]]:
    """The lines a reader takes numbers from: no code, no link targets, no image sources."""
    return [
        (number, _blank(_blank(line, MARKDOWN_LINK), HTML_REFERENCE))
        for number, line in prose_lines(text)
    ]


def number_tokens(line: str):
    """Yield (start, end, text) of the numbers of a line, leaving out names such as F1 or 9B."""
    for match in NUMBER.finditer(line):
        before = line[match.start() - 1 : match.start()]
        after = line[match.end() : match.end() + 1]
        if before and (before.isalnum() or before == "_"):
            continue
        if after and (after.isalpha() or after == "_"):
            continue
        yield match.start(), match.end(), match.group()


def _covered(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(a <= start and end <= b for a, b in spans)


README = "README.md"
CARD = "docs/model_card.md"
SIGNED = r"[-+]?[\d.]+"


def _row(label: str, skip: int) -> str:
    """Start of a pattern for the cell `skip` cells after the first one in the row `label`."""
    return rf"^\| {re.escape(label)} \|(?:[^|]*\|){{{skip}}} \*{{0,2}}"


def interval(doc, label, skip, file, path, fmt):
    """The three numbers of a table cell written `value [low, high]`."""
    start = _row(label, skip)
    value = rf"({SIGNED})\*{{0,2}} \["
    low = rf"{SIGNED}\*{{0,2}} \[({SIGNED}),"
    high = rf"{SIGNED}\*{{0,2}} \[{SIGNED}, ({SIGNED})\]"
    return [
        Cell(doc, start + value, file, f"{path}.value", fmt),
        Cell(doc, start + low, file, f"{path}.ci_low", fmt),
        Cell(doc, start + high, file, f"{path}.ci_high", fmt),
    ]


def plain(doc, label, skip, file, path, fmt):
    """The one number of a table cell written without an interval."""
    return Cell(doc, _row(label, skip) + r"([\d,]+) \|", file, path, fmt)


def screener_rows(doc, label, file, accuracy, macro_f1, gap):
    """The accuracy, macro-F1 and RMS TPR gap cells of one row of a results table."""
    return (
        interval(doc, label, 0, file, accuracy, "pct1")
        + interval(doc, label, 1, file, macro_f1, "dec3")
        + interval(doc, label, 2, file, gap, "dec3")
    )


def job_rows(label, key):
    """The cells of one job of the per-job table of the model card, main model."""
    base = f"test.per_occupation.{key}"
    return (
        [
            plain(CARD, label, 0, "metrics_deberta.json", f"{base}.n_female", "count"),
            plain(CARD, label, 1, "metrics_deberta.json", f"{base}.n_male", "count"),
        ]
        + interval(CARD, label, 2, "metrics_deberta.json", f"{base}.tpr_female", "pct1")
        + interval(CARD, label, 3, "metrics_deberta.json", f"{base}.tpr_male", "pct1")
        + interval(CARD, label, 4, "metrics_deberta.json", f"{base}.tpr_gap", "signed_points1")
    )


MAIN, BASE = "metrics_deberta.json", "metrics_baseline.json"
MITIGATION, AUDIT = "mitigation.json", "llm_audit.json"
FLIPS = "counterfactual_gender.csv"
PROBING = "probing.csv"
SCRUBBED_LOG = "training_log_scrubbed.csv"
CONDITION_A = "conditions.a.outside_showcase"
CONDITION_D = "conditions.d.outside_showcase"

# One row per number the two documents state: the document, a pattern with one group around the
# number as written, the results file, the path to the value in it and the format.
CELLS: list[Cell] = [
    # README: the opening
    Cell(
        README, r"trained such a model on ([\d,]+) real", "data_summary.json", "bios.train", "count"
    ),
    Cell(README, r"among ([\d,]+) bios it had", "data_summary.json", "bios.test", "count"),
    Cell(
        README,
        r"(\d+) of every 100 men but",
        MAIN,
        "test.per_occupation.surgeon.tpr_male.value",
        "per100",
    ),
    Cell(
        README,
        r"only (\d+) of every 100 women",
        MAIN,
        "test.per_occupation.surgeon.tpr_female.value",
        "per100",
    ),
    Cell(
        README,
        r"opposite: (\d+) of every 100 women",
        MAIN,
        "test.per_occupation.nurse.tpr_female.value",
        "per100",
    ),
    Cell(
        README,
        r"women, (\d+) of every 100 men",
        MAIN,
        "test.per_occupation.nurse.tpr_male.value",
        "per100",
    ),
    Cell(
        README,
        r"surgeon: (\d+) of 100 qualified women",
        MAIN,
        "test.per_occupation.surgeon.tpr_female.value",
        "per100",
    ),
    Cell(
        README,
        r"against (\d+) of 100 men; nurse",
        MAIN,
        "test.per_occupation.surgeon.tpr_male.value",
        "per100",
    ),
    Cell(
        README,
        r"nurse: (\d+) of 100 women",
        MAIN,
        "test.per_occupation.nurse.tpr_female.value",
        "per100",
    ),
    Cell(
        README,
        r"against (\d+) of 100 men\.",
        MAIN,
        "test.per_occupation.nurse.tpr_male.value",
        "per100",
    ),
    # README: key findings
    Cell(
        README,
        r"in (\d+) of 150 women's bios",
        "gap_decomposition.csv",
        "screener=deberta,occupation=model,word=freeones:freq_female",
        "of150",
    ),
    Cell(
        README,
        r"decision for (\d+) of every 100 bios",
        FLIPS,
        "screener=deberta,occupation=pooled,subset=all,condition=swap:flip_rate",
        "per100",
    ),
    Cell(
        README,
        r"does \((\d+) of 100\)",
        FLIPS,
        "screener=deberta,occupation=pooled,subset=all,condition=removal:flip_rate",
        "per100",
    ),
    Cell(
        README,
        r"reacts more: (\d+) against \d+ of every",
        FLIPS,
        "screener=baseline,occupation=pooled,subset=all,condition=swap:flip_rate",
        "per100",
    ),
    Cell(
        README,
        r"reacts more: \d+ against (\d+) of every",
        FLIPS,
        "screener=baseline,occupation=pooled,subset=all,condition=removal:flip_rate",
        "per100",
    ),
    Cell(
        README,
        r"for (\d+)% to \d+% of bios",
        PROBING,
        "model=deberta,layer=6,subset=all:accuracy",
        "per100",
    ),
    Cell(
        README,
        r"for \d+% to (\d+)% of bios",
        PROBING,
        "model=deberta,layer=0,subset=all:accuracy",
        "per100",
    ),
    Cell(
        README,
        r"against (\d+)% from the job alone",
        PROBING,
        "model=occupation_only,subset=all:accuracy",
        "per100",
    ),
    Cell(
        README,
        r"gender gap from ([\d.]+) to",
        MITIGATION,
        f"{CONDITION_A}.rms_tpr_gap.value",
        "dec3",
    ),
    Cell(
        README,
        r"gap from [\d.]+ to ([\d.]+) for",
        MITIGATION,
        f"{CONDITION_D}.rms_tpr_gap.value",
        "dec3",
    ),
    Cell(
        README,
        r"still passes on (\d+) more qualified",
        MITIGATION,
        f"{CONDITION_D}.tpr_gap.model.value",
        "per100",
    ),
    Cell(
        README, r"against (\d+) before", MITIGATION, f"{CONDITION_A}.tpr_gap.model.value", "per100"
    ),
    Cell(
        README, r"differs by (-[\d.]+) points", AUDIT, "statistics.primary.value", "signed_points1"
    ),
    Cell(README, r"\(interval (-[\d.]+) to", AUDIT, "statistics.primary.ci_low", "signed_points1"),
    Cell(README, r"to (\+[\d.]+)\)", AUDIT, "statistics.primary.ci_high", "signed_points1"),
    Cell(
        README,
        r"by about (\d+) points against no name",
        AUDIT,
        "statistics.pooled_names_vs_none.value",
        "abs_per100",
    ),
    Cell(
        README,
        r"changes (\d+)% and \d+% of its",
        AUDIT,
        "prompt_sensitivity.v2.flip_rate_against_v1",
        "per100",
    ),
    Cell(
        README,
        r"changes \d+% and (\d+)% of its",
        AUDIT,
        "prompt_sensitivity.v3.flip_rate_against_v1",
        "per100",
    ),
    Cell(README, r"\((\d+) against \d+ of every 100 bios\)", MAIN, "test.accuracy.value", "per100"),
    Cell(README, r"\(\d+ against (\d+) of every 100 bios\)", BASE, "test.accuracy.value", "per100"),
    # README: results and data
    Cell(README, r"Tested on ([\d,]+) bios", "data_summary.json", "bios.test", "count"),
    *screener_rows(
        README,
        "Word counts + logistic regression (baseline)",
        BASE,
        "test.accuracy",
        "test.macro_f1",
        "test.rms_tpr_gap",
    ),
    *screener_rows(
        README,
        "DeBERTa-v3-small, fine-tuned (main model)",
        MAIN,
        "test.accuracy",
        "test.macro_f1",
        "test.rms_tpr_gap",
    ),
    Cell(
        README,
        r"gives about ([\d.]+)\.",
        "gap_statistics.json",
        "main_model.rms_tpr_gap.null_mean",
        "dec2",
    ),
    Cell(README, r"on a ([\d,]+)-bio subsample", "data_summary.json", "bios.train", "count"),
    Cell(
        README,
        r"subset of ([\d,]+) test bios",
        "data_summary.json",
        "bios.showcase_subset",
        "count",
    ),
    # README: run times
    Cell(README, r"\| (\d+) minutes on the CPU", BASE, "runtime_seconds", "minutes0"),
    Cell(README, r"; ([\d.]+) hours on the GPU", MAIN, "training_seconds", "hours2"),
    Cell(
        README,
        r"run ([\d.]+) hours for the main",
        "shapley_summary.json",
        "screeners.deberta.total_seconds",
        "hours2",
    ),
    Cell(
        README,
        r"and (\d+) minutes for the baseline;",
        "shapley_summary.json",
        "screeners.baseline.total_seconds",
        "minutes0",
    ),
    Cell(
        README, r"probes (\d), \d and \d minutes", "probing.json", "seconds.references", "minutes0"
    ),
    Cell(
        README, r"probes \d, (\d) and \d minutes", "probing.json", "seconds.pretrained", "minutes0"
    ),
    Cell(README, r"probes \d, \d and (\d) minutes", "probing.json", "seconds.deberta", "minutes0"),
    Cell(README, r"patching (\d+) seconds", "patching.json", "seconds", "seconds0"),
    Cell(
        README, r"name check (\d+) minutes", AUDIT, "manipulation_check.seconds.total", "minutes0"
    ),
    Cell(README, r"pilot ([\d.]+) minutes", "llm/pilot.json", "seconds_total", "minutes1"),
    Cell(README, r"run ([\d.]+) hours;", AUDIT, "run_cost.seconds_total", "hours2"),
    # Model card: details
    Cell(
        CARD,
        r"(\d[\d,]*) bios of the training subsample",
        "data_summary.json",
        "bios.train",
        "count",
    ),
    Cell(CARD, r"; ([\d,]+) dev bios choose", "data_summary.json", "bios.dev", "count"),
    Cell(CARD, r"warm-up over (\d+)% of", MAIN, "hyperparameters.warmup_ratio", "per100"),
    Cell(CARD, r"bfloat16 autocast, (\d+) epochs", MAIN, "hyperparameters.epochs", "count"),
    Cell(
        CARD, r"effective batch of (\d+) \(", MAIN, "hyperparameters.effective_batch_size", "count"
    ),
    Cell(CARD, r"\((\d+) bios per step", MAIN, "hyperparameters.batch_size", "count"),
    Cell(CARD, r", (\d+) steps added up", MAIN, "hyperparameters.gradient_accumulation", "count"),
    Cell(CARD, r"at most (\d+) tokens per bio", MAIN, "hyperparameters.max_length", "count"),
    Cell(
        CARD, r"Learning rate (3e-5), after", MAIN, "hyperparameters.learning_rate", "learning_rate"
    ),
    Cell(CARD, r"also trying (5e-5);", MAIN, "runs.0.learning_rate", "learning_rate"),
    Cell(
        CARD, r"best dev macro-F1, at step ([\d,]+)\.", MAIN, "hyperparameters.best_step", "count"
    ),
    Cell(CARD, r"\*\* ([\d,]+) s for the chosen run", MAIN, "runs.1.training_seconds", "count"),
    Cell(CARD, r"; ([\d,]+) s for both learning rates", MAIN, "training_seconds", "count"),
    Cell(CARD, r"at least (\d+) bios, and", BASE, "hyperparameters.min_df", "count"),
    Cell(CARD, r"C is (\d+); both", BASE, "hyperparameters.C", "count"),
    Cell(CARD, r"trained in ([\d,]+) s on the CPU", BASE, "runtime_seconds", "count"),
    Cell(
        CARD,
        r"Learning rate (3e-5), \d epochs",
        SCRUBBED_LOG,
        "step=8250:base_learning_rate",
        "learning_rate",
    ),
    Cell(CARD, r"Learning rate 3e-5, (\d) epochs", SCRUBBED_LOG, "step=8250:epoch", "count"),
    Cell(CARD, r"macro-F1, ([\d.]+) at step", SCRUBBED_LOG, "step=8250:dev_macro_f1", "dec3"),
    Cell(CARD, r"macro-F1, [\d.]+ at step ([\d,]+)\.", SCRUBBED_LOG, "step=8250:step", "count"),
    # Model card: use, factors, metrics, data
    Cell(CARD, r"on ([\d,]+) web bios", "data_summary.json", "bios.train", "count"),
    Cell(CARD, r"train on ([\d,]+) training bios", "data_summary.json", "bios.train", "count"),
    Cell(
        CARD,
        r"over the test bios \(([\d,]+) resamples\)",
        "gap_statistics.json",
        "main_model.n_bootstrap",
        "count",
    ),
    Cell(
        CARD,
        r"gives a gap of about ([\d.]+)\.",
        "gap_statistics.json",
        "main_model.rms_tpr_gap.null_mean",
        "dec2",
    ),
    Cell(CARD, r"\*\*Test split\.\*\* ([\d,]+) bios", "data_summary.json", "bios.test", "count"),
    Cell(
        CARD,
        r"\*\*Showcase subset\.\*\* ([\d,]+) test bios",
        "data_summary.json",
        "bios.showcase_subset",
        "count",
    ),
    Cell(
        CARD,
        r"also scored on the ([\d,]+) test bios outside",
        MITIGATION,
        "evaluation_set.bios_scored",
        "count",
    ),
    # Model card: results tables
    *screener_rows(
        CARD, "Baseline model", BASE, "test.accuracy", "test.macro_f1", "test.rms_tpr_gap"
    ),
    *screener_rows(CARD, "Main model", MAIN, "test.accuracy", "test.macro_f1", "test.rms_tpr_gap"),
    *screener_rows(
        CARD,
        "Scrubbed model, raw bios",
        MITIGATION,
        "conditions.c.full_test.accuracy",
        "conditions.c.full_test.macro_f1",
        "conditions.c.full_test.rms_tpr_gap",
    ),
    *screener_rows(
        CARD,
        "Scrubbed model, scrubbed bios",
        MITIGATION,
        "conditions.d.full_test.accuracy",
        "conditions.d.full_test.macro_f1",
        "conditions.d.full_test.rms_tpr_gap",
    ),
    *job_rows("surgeon", "surgeon"),
    *job_rows("software engineer", "software_engineer"),
    *job_rows("composer", "composer"),
    *job_rows("dietitian", "dietitian"),
    *job_rows("nurse", "nurse"),
    *job_rows("model", "model"),
    Cell(
        CARD,
        r"passes on (\d+), and of 100 men",
        MAIN,
        "test.per_occupation.surgeon.tpr_female.value",
        "per100",
    ),
    Cell(
        CARD,
        r"of 100 men (\d+), so a recruiter",
        MAIN,
        "test.per_occupation.surgeon.tpr_male.value",
        "per100",
    ),
    Cell(
        CARD,
        r"passes on (\d+) of 100 women and",
        MAIN,
        "test.per_occupation.model.tpr_female.value",
        "per100",
    ),
    Cell(
        CARD,
        r"and (\d+) of 100 men\. Software",
        MAIN,
        "test.per_occupation.model.tpr_male.value",
        "per100",
    ),
    Cell(
        CARD, r"\(correlation ([\d.]+),", "gap_statistics.json", "main_model.pearson.value", "dec2"
    ),
    Cell(
        CARD,
        r"95% interval ([\d.]+) to",
        "gap_statistics.json",
        "main_model.pearson.fisher_ci_low",
        "dec2",
    ),
    Cell(
        CARD,
        r"to ([\d.]+) over the 28 jobs",
        "gap_statistics.json",
        "main_model.pearson.fisher_ci_high",
        "dec2",
    ),
    Cell(
        CARD, r"holds (\d+) men dietitians", MAIN, "test.per_occupation.dietitian.n_male", "count"
    ),
    # Model card: the LLM
    Cell(CARD, r"Temperature (\d),", AUDIT, "settings.temperature", "count"),
    Cell(CARD, r"presence penalty (\d),", AUDIT, "settings.presence_penalty", "count"),
    Cell(CARD, r"at most (\d+) tokens\.", AUDIT, "settings.max_tokens", "count"),
    Cell(CARD, r"\*\*What I measured\.\*\* (\d+) profiles", AUDIT, "profiles.n_profiles", "count"),
    Cell(CARD, r": (\d+) qualified candidates and", AUDIT, "profiles.n_match", "count"),
    Cell(CARD, r"and (\d+) who are not\.", AUDIT, "profiles.n_nonmatch", "count"),
    Cell(CARD, r"says yes to (\d+) with no header", AUDIT, "statistics.tpr|none.value", "per100"),
    Cell(CARD, r"\(interval (\d+) to \d+\)", AUDIT, "statistics.tpr|none.ci_low", "per100"),
    Cell(CARD, r"\(interval \d+ to (\d+)\)", AUDIT, "statistics.tpr|none.ci_high", "per100"),
    Cell(
        CARD,
        r"man's name, is (-[\d.]+) points",
        AUDIT,
        "statistics.primary.value",
        "signed_points1",
    ),
    Cell(CARD, r"\(interval (-[\d.]+) to", AUDIT, "statistics.primary.ci_low", "signed_points1"),
    Cell(CARD, r"to (\+[\d.]+)\)", AUDIT, "statistics.primary.ci_high", "signed_points1"),
    Cell(
        CARD,
        r"lowers the yes-rate by ([\d.]+) points",
        AUDIT,
        "statistics.pooled_names_vs_none.value",
        "abs_points1",
    ),
    Cell(
        CARD,
        r"\(interval ([\d.]+) to [\d.]+\), a lead",
        AUDIT,
        "statistics.pooled_names_vs_none.ci_high",
        "abs_points1",
    ),
    Cell(
        CARD,
        r"\(interval [\d.]+ to ([\d.]+)\), a lead",
        AUDIT,
        "statistics.pooled_names_vs_none.ci_low",
        "abs_points1",
    ),
    Cell(CARD, r"yes-rate to (\d+) and \d+ of 100\.", AUDIT, "prompt_sensitivity.v2.tpr", "per100"),
    Cell(CARD, r"yes-rate to \d+ and (\d+) of 100\.", AUDIT, "prompt_sensitivity.v3.tpr", "per100"),
    Cell(CARD, r"\*\*Cost\.\*\* ([\d,]+) calls", AUDIT, "run_cost.n_calls", "count"),
    Cell(CARD, r"calls in ([\d.]+) hours", AUDIT, "run_cost.seconds_total", "hours2"),
    Cell(CARD, r"(\d+) profiles that are not qualified", AUDIT, "arms.none.n_nonmatch", "count"),
]

# Numbers that are not results: names, units, years, counts in the descriptions and times that
# only docs/results.md records. `None` means every checked document.
ALLOWED: list[Allowed] = [
    Allowed(None, r"\b28\b", "the 28 occupations of Bias in Bios"),
    Allowed(None, r"\b100\b", "the base of the shares quoted as 'of every 100'"),
    Allowed(None, r"\b150\b", "the showcase subset takes up to 150 bios per job and gender"),
    Allowed(None, r"\b95%", "the level of the intervals"),
    Allowed(None, r"\bADR \d{4}\b", "the number of a decision record"),
    Allowed(None, r"\b(?:2019|2020|2021)\b", "the years of the cited works"),
    Allowed(None, r"[Ss]eed 42", "the global seed, config.SEED"),
    Allowed(None, r"\b0 would be equal treatment|\babove 0\b", "the zero of the gap"),
    Allowed(None, r"Apache-2\.0|Ternary Bonsai 2 9B|\b2-bit\b", "the identity of the LLM"),
    Allowed(None, r"(?i)RTX 3050 laptop GPU \(4 GB\)", "the laptop"),
    Allowed(README, r'width="520"', "image width"),
    Allowed(
        README, r"Python 3\.12|CUDA 12\.8", "software versions pinned in pyproject.toml and uv.lock"
    ),
    Allowed(README, r"arXiv:2111\.09543|CC BY 4\.0|01/01/20(?:22|25)", "the citations"),
    Allowed(README, r"about 3 minutes", "the fast test suite, measured on the laptop"),
    Allowed(README, r"about 30 seconds", "gap_decomposition.py, docs/results.md"),
    Allowed(README, r"about 45 seconds", "counterfactual_gender.py, docs/results.md"),
    Allowed(
        README,
        r"about 6 minutes|about 10 and 15 minutes",
        "deletion curves and reruns, docs/results.md",
    ),
    Allowed(README, r"about 1 minute each", "probing_features.py, docs/results.md"),
    Allowed(README, r"about 1\.9 hours", "train_scrubbed.py, docs/results.md"),
    Allowed(
        README,
        r"about 2\.8 minutes per condition and 2 minutes for",
        "mitigation.py, docs/results.md",
    ),
    Allowed(README, r"about 25 seconds", "llm_analyse.py, docs/results.md"),
]


def check_numbers(
    root: Path,
    cells: Sequence[Cell] = CELLS,
    allowed: Sequence[Allowed] = ALLOWED,
    documents: Sequence[str] = NUMBER_DOCUMENTS,
) -> list[str]:
    """One message per number that differs from results/ and per number no row or phrase covers."""
    errors = [
        f"{c.document}: a row of the table names a document that is not checked"
        for c in cells
        if c.document not in documents
    ]
    for name in documents:
        path = root / name
        if not path.is_file():
            errors.append(f"{name}: document is missing")
            continue
        lines = claim_lines(path.read_text(encoding="utf-8"))
        spans: dict[int, list[tuple[int, int]]] = {}
        for cell in (c for c in cells if c.document == name):
            pattern = re.compile(cell.pattern)
            found = [(n, m) for n, line in lines for m in pattern.finditer(line)]
            if not found:
                errors.append(f"{name}: no text matches {cell.pattern!r}")
                continue
            for number, match in found:
                spans.setdefault(number, []).append(match.span(1))
            try:
                expected = FORMATS[cell.fmt](read_value(root / "results", cell.file, cell.path))
            except (OSError, KeyError, ValueError, IndexError, TypeError) as exc:
                errors.append(f"{name}: cannot read {cell.file} {cell.path}: {exc!r}")
                continue
            for number, match in found:
                if match.group(1) != expected:
                    errors.append(
                        f"{name}:{number}: {match.group(1)} in the text, {expected} in "
                        f"{cell.file} {cell.path}"
                    )
        for entry in (a for a in allowed if a.document in (None, name)):
            pattern = re.compile(entry.pattern)
            for number, line in lines:
                spans.setdefault(number, []).extend(m.span() for m in pattern.finditer(line))
        for number, line in lines:
            for start, end, token in number_tokens(line):
                if not _covered(spans.get(number, []), start, end):
                    errors.append(
                        f"{name}:{number}: number {token} is covered by no row or allowed phrase"
                    )
    return errors


def main(root: Path = REPO_ROOT) -> int:
    errors = check_links(root) + check_numbers(root)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
