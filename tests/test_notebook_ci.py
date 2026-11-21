import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED = re.compile(r"^(nbmake|ipykernel)==[0-9][0-9A-Za-z.]*$")


def _dev_pins():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {spec for spec in pyproject["dependency-groups"]["dev"] if PINNED.match(spec)}


def test_the_dev_group_pins_nbmake_and_ipykernel():
    assert {spec.split("==")[0] for spec in _dev_pins()} == {"nbmake", "ipykernel"}


def test_ci_installs_the_same_pins():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    install = re.search(r"pip install -e \.[^\n]*", ci).group(0)
    pins = _dev_pins()
    assert pins
    assert pins <= set(install.split())


def test_ci_runs_the_notebook():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert re.search(r"^\s*- run: pytest --nbmake notebooks/audit\.ipynb$", ci, re.MULTILINE)


def test_the_readme_points_to_the_notebook():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "notebooks/audit.ipynb" in readme
