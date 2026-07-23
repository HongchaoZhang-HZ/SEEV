"""Contract tests for the site CI gate and the GitHub Pages workflow.

Both workflows are validated as plain text (no YAML parser dependency), matching
the convention in ``tests/ci``. The pull-request documentation gate is a stable
``site`` job in ``.github/workflows/ci.yml`` that runs the strict Sphinx build
on the workflow's existing pull-request and ``main`` triggers. The Pages
workflow in ``.github/workflows/pages.yml`` retains its own separate ``site``
build job and ``deploy`` job. These checks assert the stable job IDs and display
names, Python 3.10, read-only build permission, the strict Sphinx build command,
the triggers, and the official upload/configure/deploy steps with the correct
job-scoped Pages permissions.
"""

import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_WORKFLOWS = os.path.join(_ROOT, ".github", "workflows")
_PAGES = os.path.join(_WORKFLOWS, "pages.yml")
_CI = os.path.join(_WORKFLOWS, "ci.yml")


def _read(path):
    assert os.path.isfile(path), f"missing workflow: {path}"
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _run_commands(text):
    return re.findall(r"^\s*run:\s*(.+?)\s*$", text, re.MULTILINE)


def _uses(text):
    return re.findall(r"^\s*uses:\s*(.+?)\s*$", text, re.MULTILINE)


def test_pages_workflow_exists():
    assert os.path.isfile(_PAGES)


def test_pages_triggers_on_main_and_manual_dispatch():
    text = _read(_PAGES)
    assert re.search(r"^on:", text, re.MULTILINE)
    assert re.search(r"^\s*push:", text, re.MULTILINE)
    assert re.search(r"-\s*main\b", text)
    assert re.search(r"^\s*workflow_dispatch:", text, re.MULTILINE)


def test_pages_defines_stable_site_and_deploy_jobs():
    text = _read(_PAGES)
    jobs_text = text[text.index("jobs:") :]
    job_ids = re.findall(r"^  ([A-Za-z0-9_-]+):\s*$", jobs_text, re.MULTILINE)
    assert job_ids == ["site", "deploy"]
    assert re.search(r"^\s*name:\s*site\s*$", text, re.MULTILINE)


def test_site_job_builds_with_python_310_and_strict_sphinx():
    text = _read(_PAGES)
    assert re.search(r'python-version:\s*"?3\.10"?', text)
    assert re.search(r"^\s*timeout-minutes:\s*\d+", text, re.MULTILINE)
    commands = _run_commands(text)
    assert "python -m pip install -r requirements-docs.txt" in commands
    assert (
        "python -m sphinx -W --keep-going -b html site site/_build/html" in commands
    )


def test_site_job_has_read_only_contents_permission():
    text = _read(_PAGES)
    # The build job must declare least-privilege read-only contents access.
    assert re.search(r"^\s*contents:\s*read\s*$", text, re.MULTILINE)


def test_pages_uses_official_actions_and_deploys_built_html_only():
    text = _read(_PAGES)
    uses = " ".join(_uses(text))
    assert "actions/checkout@v7" in uses
    assert "actions/setup-python@v7" in uses
    assert "actions/configure-pages@v6" in uses
    assert "actions/upload-pages-artifact@v5" in uses
    assert "actions/deploy-pages@v5" in uses
    assert re.search(r"path:\s*site/_build/html\s*$", text, re.MULTILINE)


def test_deploy_job_has_pages_and_idtoken_permissions_and_environment():
    text = _read(_PAGES)
    deploy_text = text[text.index("\n  deploy:") :]
    assert re.search(r"^\s*pages:\s*write\s*$", deploy_text, re.MULTILINE)
    assert re.search(r"^\s*id-token:\s*write\s*$", deploy_text, re.MULTILINE)
    assert re.search(r"^\s*name:\s*github-pages\s*$", deploy_text, re.MULTILINE)
    assert "actions/configure-pages@v6" in deploy_text


def test_ci_runs_on_pull_request_and_main():
    text = _read(_CI)
    assert re.search(r"^\s*pull_request:", text, re.MULTILINE)
    assert re.search(r"^\s*push:", text, re.MULTILINE)
    assert re.search(r"-\s*main\b", text)


def test_ci_defines_both_unit_and_site_jobs():
    text = _read(_CI)
    jobs_text = text[text.index("jobs:") :]
    job_ids = re.findall(r"^  ([A-Za-z0-9_-]+):\s*$", jobs_text, re.MULTILINE)
    assert job_ids == ["unit", "site"]
    assert re.search(r"^\s*name:\s*site\s*$", text, re.MULTILINE)


def test_ci_site_gate_builds_with_python_310_and_strict_sphinx():
    text = _read(_CI)
    # Isolate the ``site`` job so the assertions cannot be satisfied by the
    # ``unit`` job above it.
    site_text = text[text.index("\n  site:") :]
    assert re.search(r'python-version:\s*"?3\.10"?', site_text)
    assert re.search(r"^\s*timeout-minutes:\s*\d+", site_text, re.MULTILINE)
    assert re.search(r"^\s*contents:\s*read\s*$", site_text, re.MULTILINE)
    commands = _run_commands(site_text)
    assert "python -m pip install -r requirements-docs.txt" in commands
    assert (
        "python -m sphinx -W --keep-going -b html site site/_build/html" in commands
    )


def test_existing_unit_ci_job_is_preserved():
    text = _read(_CI)
    assert re.search(r"^\s*name:\s*unit\s*$", text, re.MULTILINE)
    commands = _run_commands(text)
    assert "python -m pytest tests/unit tests/ci -q --junitxml=unit.xml" in commands
