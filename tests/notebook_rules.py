"""Rules the audit notebook must obey, checked on its parsed JSON.

Every number a reader sees in prose has to come from a result file, so prose may hold no digits
except the few listed in `ALLOWED`. A cell source is Markdown text or Python code; for code the
rule looks at the string literals and at the numbers inside f-string expressions.
"""

import ast
import re

ALLOWED = re.compile(
    r"[\w./-]+\.(?:py|csv|json|jsonl|png|pdf|ipynb|md)\b"  # file names such as figure_f3.py
    r"|(?<![\w-])F[1-8]\b"  # figure names
    r"|\bADR \d{4}\b"  # decision record numbers
    r"|\b[Mm]acro-F1\b"  # the metric
    r"|\bDeBERTa-v3-small\b"  # the main model
    r"|(?:(?<=\()|(?<=, ))(?:19|20)\d{2}(?=[);])"  # a year in a citation
)
NUMBER_WORDS = (
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty"
    r"|sixty|seventy|eighty|ninety|dozen|hundred|thousand|million|half|twice|double|triple|quarter)\b"
)
SCALES = {0, 1, 100}
READERS = {"read_json", "read_csv", "show_figure"}
PROSE_CALLS = {"say", "Markdown"}
FILE_LIKE = re.compile(r"[\w./-]+\.(?:json|jsonl|csv|png)")


def source(cell):
    text = cell["source"]
    return "".join(text) if isinstance(text, list) else text


def loose_numbers(text):
    """The words of `text` that hold a digit or are a number in words, bar the allowed tokens."""
    return re.findall(rf"\S*\d\S*|{NUMBER_WORDS}", ALLOWED.sub(" ", text), re.IGNORECASE)


def _literals(node):
    """The string pieces that are written out in an argument: not those inside {expressions}."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node
    elif isinstance(node, ast.JoinedStr):
        for part in node.values:
            yield from _literals(part)
    elif isinstance(node, ast.BinOp):
        yield from _literals(node.left)
        yield from _literals(node.right)


def _fragments(tree):
    """(line, column, token) for every typed number in the code of one cell."""
    in_spec = {
        id(inner)
        for node in ast.walk(tree)
        if isinstance(node, ast.FormattedValue) and node.format_spec
        for inner in ast.walk(node.format_spec)
    }
    in_prose = {
        id(literal)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in PROSE_CALLS
        for argument in node.args
        for literal in _literals(argument)
    }
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in in_spec
            and (id(node) in in_prose or any(c.isspace() for c in node.value))
        ):
            found += [(node.lineno, node.col_offset, t) for t in loose_numbers(node.value)]
        if isinstance(node, ast.FormattedValue):
            for inner in ast.walk(node.value):
                if _is_number(inner) and inner.value not in SCALES:
                    found.append((inner.lineno, inner.col_offset, str(inner.value)))
    return found


def _is_number(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, int | float)


def hand_typed_numbers(nb):
    """(cell index, token) for every number typed into prose, in reading order."""
    found = []
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            found += [(index, t) for t in loose_numbers(source(cell))]
        elif cell["cell_type"] == "code":
            tree = ast.parse(source(cell))
            found += [(index, t) for _, _, t in sorted(_fragments(tree), key=lambda f: f[:2])]
    return found


def _calls(nb, names):
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse(source(cell))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in names
            ):
                yield index, node


def result_reads(nb):
    """The file names the notebook reads, in order; each must be written as a string literal."""
    names = []
    for _, call in sorted(
        _calls(nb, READERS), key=lambda pair: (pair[0], pair[1].lineno, pair[1].col_offset)
    ):
        first = call.args[0] if call.args else None
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            raise ValueError(f"line {call.lineno}: the file name must be a string literal")
        names.append(first.value)
    return names


def prints(nb):
    """The cells that call print; all prose goes through Markdown."""
    return sorted({index for index, _ in _calls(nb, {"print"})})


def result_literals(nb):
    """Every string in the code that names a result file, whether or not a reader opens it."""
    names = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            for node in ast.walk(ast.parse(source(cell))):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and FILE_LIKE.fullmatch(node.value)
                ):
                    names.append(node.value)
    return names
