import pytest

from hiring_bias_xai import ine_names


def _first_name_sheet(countries: dict) -> list[list]:
    """A sheet in the layout of `nombres_por_nacionalidades.xls`: one block of three columns per
    country, the men's top list from row 5 and the women's from row 20."""
    grid = [[""] * (1 + 3 * len(countries)) for _ in range(30)]
    for k, (country, names) in enumerate(countries.items()):
        col = 1 + 3 * k
        for top, sex, label in ((3, "male", "hombres"), (18, "female", "mujeres")):
            grid[top][col] = country
            grid[top + 1][col] = f"NOMBRE ({label})"
            for i, name in enumerate(names[sex]):
                grid[top + 2 + i][col] = name
                grid[top + 2 + i][col + 1] = 1000.0 - i
    return grid


def _surname_sheet(countries: dict) -> list[list]:
    grid = [[""] * (1 + 3 * len(countries)) for _ in range(25)]
    for k, (country, surnames) in enumerate(countries.items()):
        col = 1 + 3 * k
        grid[3][col] = country
        grid[4][col] = "PRIMER APELLIDO"
        for i, surname in enumerate(surnames):
            grid[5 + i][col] = surname
    return grid


def _spain_sheet(label: str, names: list[str]) -> list[list]:
    grid = [["Nombres más frecuentes"], [], [], [None, "ESPAÑA"], [None, label]]
    return grid + [[i + 1.0, name, 1000.0 - i] for i, name in enumerate(names)]


def _files(**extra):
    first = {
        "EUROPA": _first_name_sheet(
            {
                "REINO UNIDO": {
                    "male": ["JOHN", "DAVID", "PAUL"],
                    "female": ["SUSAN", "ANN", "JANE"],
                },
                "RUMANÍA": {
                    "male": ["ION", "MIHAI", "ANDREI"],
                    "female": ["MARIA", "ELENA", "ANA"],
                },
            }
        ),
        "ÁFRICA": _first_name_sheet(
            {
                "MARRUECOS": {
                    "male": ["MOHAMED", "AHMED", "ALI"],
                    "female": ["FATIMA", "KHADIJA", "AICHA"],
                }
            }
        ),
        "AMÉRICA": _first_name_sheet(
            {
                "COLOMBIA": {"male": ["JUAN", "CARLOS", "LUIS"], "female": ["LUZ", "ANA", "DIANA"]},
                "ECUADOR": {
                    "male": ["JOSE", "CARLOS", "WILSON"],
                    "female": ["MARIA", "ROSA", "ANA"],
                },
                "VENEZUELA": {
                    "male": ["LUIS", "ANGEL", "JESUS"],
                    "female": ["YOLANDA", "ROSA", "NORELIS"],
                },
            }
        ),
    }
    surnames = {
        "EUROPA": _surname_sheet(
            {"REINO UNIDO": ["SMITH", "JONES", "TAYLOR"], "RUMANÍA": ["POPESCU", "IONESCU", "POPA"]}
        ),
        "ÁFRICA": _surname_sheet({"MARRUECOS": ["EL AMRANI", "BENALI", "MOUSSA"]}),
        "AMÉRICA": _surname_sheet(
            {
                "COLOMBIA": ["RODRIGUEZ", "GOMEZ", "CASTRO"],
                "ECUADOR": ["MORALES", "CHAVEZ", "CASTRO"],
                "VENEZUELA": ["PEREZ", "GONZALEZ", "SALAS"],
            }
        ),
    }
    spain_first = {
        "ESPAÑA_100_hombres": _spain_sheet("NOMBRE COMPLETO", ["ANTONIO", "MANUEL", "JOSE"]),
        "ESPAÑA_100_mujeres": _spain_sheet("NOMBRE COMPLETO", ["MARIA CARMEN", "MARIA", "CARMEN"]),
    }
    spain_surnames = {
        "ESPAÑA_100": _spain_sheet(
            "PRIMER APELLIDO", ["GARCIA", "RODRIGUEZ", "GONZALEZ", "PEREZ", "GOMEZ"]
        )
    }
    return {
        "first_names": first,
        "surnames": surnames,
        "spain_first_names": spain_first,
        "spain_surnames": spain_surnames,
    } | extra


def test_each_group_gets_its_top_first_names_per_sex_and_surnames_in_title_case():
    groups = ine_names.build_groups(_files(), n_first=2, n_surnames=2)["groups"]
    assert list(groups) == ["Spain", "Morocco", "Romania", "Hispanic America", "United Kingdom"]
    assert groups["Romania"] == {
        "first_names": {"female": ["Maria", "Elena"], "male": ["Ion", "Mihai"]},
        "surnames": ["Popescu", "Ionescu"],
    }
    assert groups["United Kingdom"]["first_names"]["male"] == ["John", "David"]
    assert groups["Morocco"]["surnames"] == ["El Amrani", "Benali"]
    assert groups["Spain"] == {
        "first_names": {"female": ["Maria Carmen", "Maria"], "male": ["Antonio", "Manuel"]},
        "surnames": ["Garcia", "Rodriguez"],
    }


def test_hispanic_america_pools_its_three_countries_and_drops_surnames_that_are_spanish():
    built = ine_names.build_groups(_files(), n_first=2, n_surnames=3)
    hispanic = built["groups"]["Hispanic America"]
    assert hispanic["first_names"]["male"] == ["Juan", "Jose", "Luis", "Carlos", "Angel"]
    assert hispanic["first_names"]["female"] == ["Luz", "Maria", "Yolanda", "Ana", "Rosa"]
    # Rodriguez, Gomez, Perez and Gonzalez are in the Spanish top list; Castro is listed twice.
    assert hispanic["surnames"] == ["Morales", "Chavez", "Castro", "Salas"]
    assert built["dropped_surnames"] == ["Rodriguez", "Perez", "Gomez", "Gonzalez"]


def test_a_country_missing_from_the_files_stops_the_build_and_names_it():
    files = _files()
    del files["first_names"]["ÁFRICA"]
    with pytest.raises(ValueError, match="MARRUECOS"):
        ine_names.build_groups(files, n_first=2, n_surnames=2)


class _Sheet:
    def __init__(self, name, rows):
        self.name = name
        self._rows = rows
        self.nrows = len(rows)

    def row_values(self, index):
        return self._rows[index]


class _Workbook:
    def __init__(self, sheets):
        self._sheets = sheets

    def sheets(self):
        return self._sheets


def test_the_workbook_is_read_from_its_bytes_into_one_grid_of_cells_per_sheet(monkeypatch):
    seen = []

    def fake_open(**kwargs):
        seen.append(kwargs)
        return _Workbook([_Sheet("EUROPA", [["a", 1.0], ["b", ""]]), _Sheet("ASIA", [])])

    monkeypatch.setattr(ine_names.xlrd, "open_workbook", fake_open)
    assert ine_names.read_sheets(b"bytes") == {"EUROPA": [["a", 1.0], ["b", ""]], "ASIA": []}
    assert seen == [{"file_contents": b"bytes", "formatting_info": False}]


def test_bytes_that_are_not_a_workbook_raise_a_value_error():
    with pytest.raises(ValueError, match="not an .xls workbook"):
        ine_names.read_sheets(b"MZ this is not a spreadsheet")


def test_a_country_listed_twice_under_the_same_heading_stops_the_build():
    files = _files()
    files["first_names"]["OTRA"] = files["first_names"]["ÁFRICA"]
    with pytest.raises(ValueError, match="MARRUECOS"):
        ine_names.build_groups(files, n_first=2, n_surnames=2)
