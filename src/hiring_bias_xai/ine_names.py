"""First names and surnames by origin group, from the INE (Spain's statistics office) lists.

The files are data, never code: they are checked against a pinned hash before anything reads
them and are only read cell by cell as text and numbers.
"""

import unicodedata

N_FIRST_NAMES = 10  # per sex
N_SURNAMES = 20

# group -> countries pooled in it. Spain comes from the files for the whole of Spain.
NATIONALITIES = {
    "Morocco": ("MARRUECOS",),
    "Romania": ("RUMANIA",),
    "Hispanic America": ("COLOMBIA", "ECUADOR", "VENEZUELA"),
    "United Kingdom": ("REINO UNIDO",),
}
GROUPS = ("Spain", *NATIONALITIES)

_SEXES = {"male": "hombres", "female": "mujeres"}


def _plain(cell) -> str:
    """The text of a cell without accents, upper case and stripped; "" for a number."""
    if not isinstance(cell, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", cell)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).upper().strip()


def _list_below(grid: list[list], row: int, col: int, n: int) -> list[str]:
    """The first `n` names in column `col` below `row`, up to the first empty cell."""
    names = []
    for line in grid[row + 1 :]:
        name = line[col].strip() if col < len(line) and isinstance(line[col], str) else ""
        if not name or len(names) == n:
            break
        names.append(name.title())
    return names


def _country_list(sheets: dict, country: str, label: str, n: int) -> list[str]:
    """The top `n` entries under the header `label` of the block headed `country`."""
    found = [
        (grid, r, c)
        for grid in sheets.values()
        for r, line in enumerate(grid[:-1])
        for c, cell in enumerate(line)
        if _plain(cell) == country and _plain(grid[r + 1][c]).startswith(label.upper())
    ]
    if len(found) != 1:
        raise ValueError(f"{len(found)} blocks of {country} under {label!r}, expected one")
    grid, r, c = found[0]
    return _list_below(grid, r + 1, c, n)


def _spanish_list(sheets: dict, sheet: str, label: str, n: int | None) -> list[str]:
    """The top `n` entries (all if None) of the whole-Spain sheet `sheet`, under `label`."""
    (grid,) = (g for name, g in sheets.items() if _plain(name) == sheet)
    for r, line in enumerate(grid):
        for c, cell in enumerate(line):
            if _plain(cell).startswith(label.upper()):
                return _list_below(grid, r, c, n if n is not None else len(grid))
    raise ValueError(f"no column {label!r} in {sheet}")


def _pooled(lists: list[list[str]]) -> list[str]:
    """The lists merged rank by rank, first country first, each name once."""
    merged: list[str] = []
    for rank in range(max(map(len, lists))):
        for names in lists:
            if rank < len(names) and names[rank] not in merged:
                merged.append(names[rank])
    return merged


def build_groups(files: dict, n_first: int = N_FIRST_NAMES, n_surnames: int = N_SURNAMES) -> dict:
    """The five origin groups with their top first names per sex and their frequent surnames.

    `files` maps `first_names`, `surnames`, `spain_first_names` and `spain_surnames` to the
    sheets of that file (name -> grid of cells). Hispanic America pools its countries rank by
    rank. Its surnames that are also among the Spanish list (all rows of `spain_surnames`) are
    dropped and returned as `dropped_surnames`. Raises ValueError if a country is missing.
    """
    spain_first, spain_surnames = files["spain_first_names"], files["spain_surnames"]
    groups = {
        "Spain": {
            "first_names": {
                sex: _spanish_list(
                    spain_first, f"ESPANA_100_{spanish.upper()}", "NOMBRE COMPLETO", n_first
                )
                for sex, spanish in _SEXES.items()
            },
            "surnames": _spanish_list(spain_surnames, "ESPANA_100", "PRIMER APELLIDO", n_surnames),
        }
    }
    for group, countries in NATIONALITIES.items():
        groups[group] = {
            "first_names": {
                sex: _pooled(
                    [
                        _country_list(files["first_names"], c, f"NOMBRE ({spanish})", n_first)
                        for c in countries
                    ]
                )
                for sex, spanish in _SEXES.items()
            },
            "surnames": _pooled(
                [
                    _country_list(files["surnames"], c, "PRIMER APELLIDO", n_surnames)
                    for c in countries
                ]
            ),
        }
    spanish = set(_spanish_list(spain_surnames, "ESPANA_100", "PRIMER APELLIDO", None))
    hispanic = groups["Hispanic America"]["surnames"]
    groups["Hispanic America"]["surnames"] = [s for s in hispanic if s not in spanish]
    return {"groups": groups, "dropped_surnames": [s for s in hispanic if s in spanish]}
