"""Contract test for the pull-request CI workflow.

The workflow is validated by parsing ``.github/workflows/ci.yml`` as plain text
so the focused test surface never needs a YAML parser dependency. The checks
assert the stable job name, triggers, permissions, Python version, dependency
install, test invocation, and the two strict CI checkers.
"""

import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_WORKFLOW = os.path.join(_ROOT, ".github", "workflows", "ci.yml")


def _workflow_text():
    assert os.path.isfile(_WORKFLOW), f"missing CI workflow: {_WORKFLOW}"
    with open(_WORKFLOW, "r", encoding="utf-8") as handle:
        return handle.read()


def _workflow_commands(text):
    return re.findall(r"^\s*run:\s*(.+?)\s*$", text, re.MULTILINE)


def test_workflow_file_exists():
    assert os.path.isfile(_WORKFLOW)


def test_triggers_pull_request_and_main():
    text = _workflow_text()
    assert re.search(r"^on:", text, re.MULTILINE)
    assert re.search(r"^\s*pull_request:", text, re.MULTILINE)
    assert re.search(r"^\s*push:", text, re.MULTILINE)
    assert re.search(r"-\s*main\b", text)


def test_contents_permission_is_read_only():
    text = _workflow_text()
    assert re.search(r"^\s*contents:\s*read\s*$", text, re.MULTILINE)


def test_unit_and_site_jobs_with_timeout_and_python_310():
    text = _workflow_text()
    jobs_text = text[text.index("jobs:") :]
    # The workflow now runs both the focused ``unit`` gate and the ``site``
    # documentation-build gate; both must be present, in this order.
    assert re.findall(r"^  ([A-Za-z0-9_-]+):\s*$", jobs_text, re.MULTILINE) == [
        "unit",
        "site",
    ]
    assert re.search(r"^\s*name:\s*unit\s*$", text, re.MULTILINE)
    assert re.search(r"^\s*name:\s*site\s*$", text, re.MULTILINE)
    assert re.search(r"^\s*timeout-minutes:\s*\d+", text, re.MULTILINE)
    assert re.search(r'python-version:\s*"?3\.10"?', text)


def test_installs_requirements_ci():
    commands = _workflow_commands(_workflow_text())
    assert "python -m pip install -r requirements-ci.txt" in commands


def test_runs_focused_suites_with_junit_xml():
    commands = _workflow_commands(_workflow_text())
    assert (
        "python -m pytest tests/unit tests/ci -q --junitxml=unit.xml"
    ) in commands


def test_junit_report_path_needs_no_precreated_directory():
    commands = _workflow_commands(_workflow_text())
    pytest_command = next(command for command in commands if " -m pytest " in command)
    assert "--junitxml=unit.xml" in pytest_command


def test_runs_both_strict_checkers():
    commands = _workflow_commands(_workflow_text())
    assert "python scripts/ci/check_test_report.py unit.xml" in commands
    assert "python scripts/ci/check_pillow.py" in commands
