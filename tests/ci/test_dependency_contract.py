"""Dependency-boundary contracts for maintained, optional, and legacy paths."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _lines(name):
    return {
        line.strip()
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _normalized_names(name):
    names = set()
    for line in _lines(name):
        if line.startswith("-r "):
            continue
        package = line.split(";", 1)[0]
        for separator in ("==", ">=", "<=", "~=", ">", "<", "["):
            package = package.split(separator, 1)[0]
        names.add(package.strip().lower().replace("-", "_"))
    return names


def test_focused_ci_excludes_optional_tooling_and_legacy_integrations():
    names = _normalized_names("requirements-ci.txt")
    assert names.isdisjoint(
        {
            "auto_lirpa",
            "brunette",
            "flake8",
            "gurobipy",
            "mypy",
            "pytest_benchmark",
            "tomlkit",
        }
    )


def test_research_runtime_excludes_dev_and_optional_legacy_extras():
    names = _normalized_names("requirements.txt")
    assert names.isdisjoint(
        {
            "auto_lirpa",
            "brunette",
            "flake8",
            "gurobipy",
            "mypy",
            "pytest",
            "tomlkit",
        }
    )


def test_legacy_integrations_are_isolated_and_attributed():
    legacy = _lines("requirements-legacy.txt")
    assert "auto_LiRPA" in legacy
    assert "gurobipy==9.1.2" in legacy
    text = (ROOT / "requirements-legacy.txt").read_text(encoding="utf-8")
    assert "exactverif-reluncbf-nips23" in text
    assert "adapted neural_clbf" in text


def test_benchmark_and_development_tools_extend_the_focused_environment():
    benchmark = _lines("requirements-benchmark.txt")
    development = _lines("requirements-dev.txt")
    assert "-r requirements-ci.txt" in benchmark
    assert "pytest-benchmark>=4.0" in benchmark
    assert "-r requirements-ci.txt" in development
    assert {"mypy", "flake8", "brunette"}.issubset(development)


def test_production_sources_do_not_import_removed_tooling():
    search = (ROOT / "EEV" / "EEV" / "Scripts" / "Search.py").read_text(
        encoding="utf-8"
    )
    verification = (
        ROOT / "EEV" / "EEV" / "Verifier" / "Verification.py"
    ).read_text(encoding="utf-8")
    assert "from tomlkit import item" not in search
    assert "from pytest import fail" not in verification
