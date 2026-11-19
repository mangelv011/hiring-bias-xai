from pathlib import Path

from hiring_bias_xai import config


def test_artifacts_dir_defaults_to_user_cache(monkeypatch):
    monkeypatch.delenv("HBX_ARTIFACTS", raising=False)
    assert config.artifacts_dir() == Path.home() / ".cache" / "hiring-bias-xai"


def test_artifacts_dir_reads_env_var_and_expands_home(monkeypatch):
    monkeypatch.setenv("HBX_ARTIFACTS", "~/somewhere")
    assert config.artifacts_dir() == Path.home() / "somewhere"


def test_results_paths_live_in_the_repo():
    repo_root = Path(__file__).resolve().parents[1]
    assert config.RESULTS_DIR == repo_root / "results"
    assert config.FIGURES_DIR == repo_root / "results" / "figures"
