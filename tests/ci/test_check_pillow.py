"""Acceptance and rejection tests for the Pillow policy checker."""

import pytest
from packaging.version import Version

import check_pillow


@pytest.mark.parametrize(
    "constraint",
    [
        "Pillow>=12.3.0",
        "pillow==12.3.0",
        "Pillow~=12.3",
        "Pillow>12.3.0",
        "Pillow[tests]>=12.3.0; python_version >= '3.10'",
    ],
)
def test_accepts_secure_direct_constraints(constraint):
    assert check_pillow.line_is_compliant(constraint)


@pytest.mark.parametrize(
    "constraint",
    [
        "Pillow",
        "Pillow<=12.3.0",
        "Pillow==9.5.0",
        "Pillow>=12.2.9",
        "Pillow>=12.3.0rc1",
        "Pillow @ https://example.invalid/pillow.whl",
    ],
)
def test_rejects_constraints_that_can_resolve_below_floor(constraint):
    assert not check_pillow.line_is_compliant(constraint)


def test_package_match_is_case_insensitive_and_exact():
    assert check_pillow.is_pillow_line("pIlLoW>=12.3.0")
    assert not check_pillow.is_pillow_line("pillow-extra>=12.3.0")
    assert not check_pillow.is_pillow_line("# Pillow==9.5.0")


def test_check_file_accepts_and_counts_direct_constraint(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text("pytest>=8\nPillow>=12.3.0\n", encoding="utf-8")
    assert check_pillow.check_file(path) == 1


def test_check_file_rejects_old_constraint(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text("Pillow==9.5.0\n", encoding="utf-8")
    with pytest.raises(check_pillow.PillowError, match="non-compliant"):
        check_pillow.check_file(path)


def test_check_file_rejects_missing_file(tmp_path):
    with pytest.raises(check_pillow.PillowError, match="not found"):
        check_pillow.check_file(tmp_path / "requirements.txt")


def test_installed_version_at_floor_passes(monkeypatch):
    monkeypatch.setattr(
        check_pillow, "installed_version", lambda: Version("12.3.0")
    )
    assert check_pillow.check_installed() == Version("12.3.0")


def test_installed_version_below_floor_fails(monkeypatch):
    monkeypatch.setattr(
        check_pillow, "installed_version", lambda: Version("12.2.9")
    )
    with pytest.raises(check_pillow.PillowError, match="below required"):
        check_pillow.check_installed()


def test_repo_check_covers_both_maintained_files(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("Pillow>=12.3.0\n", encoding="utf-8")
    (tmp_path / "requirements-ci.txt").write_text(
        "Pillow>=12.3.0\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        check_pillow, "installed_version", lambda: Version("12.3.1")
    )
    result = check_pillow.check_repo(tmp_path)
    assert result == {"installed": Version("12.3.1"), "constraints": 2}


def test_repo_check_rejects_when_no_direct_constraint_exists(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("pytest>=8\n", encoding="utf-8")
    (tmp_path / "requirements-ci.txt").write_text("torch>=2.2\n", encoding="utf-8")
    monkeypatch.setattr(
        check_pillow, "installed_version", lambda: Version("12.3.0")
    )
    with pytest.raises(check_pillow.PillowError, match="no direct"):
        check_pillow.check_repo(tmp_path)


def test_cli_returns_success_for_compliant_repo(tmp_path, monkeypatch, capsys):
    (tmp_path / "requirements.txt").write_text("Pillow>=12.3.0\n", encoding="utf-8")
    (tmp_path / "requirements-ci.txt").write_text(
        "Pillow>=12.3.0\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        check_pillow, "installed_version", lambda: Version("12.3.0")
    )
    assert check_pillow.main(["--root", str(tmp_path)]) == 0
    assert "ACCEPT:" in capsys.readouterr().out


def test_cli_returns_failure_for_noncompliant_repo(tmp_path, monkeypatch, capsys):
    (tmp_path / "requirements.txt").write_text("Pillow==9.5.0\n", encoding="utf-8")
    (tmp_path / "requirements-ci.txt").write_text(
        "Pillow>=12.3.0\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        check_pillow, "installed_version", lambda: Version("12.3.0")
    )
    assert check_pillow.main(["--root", str(tmp_path)]) == 1
    assert "REJECT:" in capsys.readouterr().err
