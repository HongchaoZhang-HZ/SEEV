"""Acceptance and rejection tests for the JUnit integrity checker."""

from xml.etree import ElementTree as ET

import pytest

import check_test_report


def _report(
    tmp_path,
    *,
    tests=20,
    failures=0,
    errors=0,
    skipped=0,
    declared_tests=None,
):
    path = tmp_path / "report.xml"
    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root,
        "testsuite",
        tests=str(tests if declared_tests is None else declared_tests),
        failures=str(failures),
        errors=str(errors),
        skipped=str(skipped),
    )
    for index in range(tests):
        case = ET.SubElement(suite, "testcase", name=f"test_{index}")
        if index < failures:
            ET.SubElement(case, "failure")
        elif index < failures + errors:
            ET.SubElement(case, "error")
        elif index < failures + errors + skipped:
            ET.SubElement(case, "skipped")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def test_accepts_report_at_recorded_floor(tmp_path):
    totals = check_test_report.check_report(_report(tmp_path))
    assert totals == {"tests": 20, "failures": 0, "errors": 0, "skipped": 0}


def test_accepts_report_above_recorded_floor(tmp_path):
    assert check_test_report.check_report(_report(tmp_path, tests=21))["tests"] == 21


def test_rejects_missing_report(tmp_path):
    with pytest.raises(check_test_report.ReportError, match="not found"):
        check_test_report.check_report(tmp_path / "missing.xml")


def test_rejects_malformed_xml(tmp_path):
    path = tmp_path / "report.xml"
    path.write_text("<testsuites><testsuite>", encoding="utf-8")
    with pytest.raises(check_test_report.ReportError, match="malformed"):
        check_test_report.check_report(path)


def test_rejects_unexpected_root_element(tmp_path):
    path = tmp_path / "report.xml"
    path.write_text("<report/>", encoding="utf-8")
    with pytest.raises(check_test_report.ReportError, match="unexpected"):
        check_test_report.check_report(path)


def test_rejects_failure(tmp_path):
    with pytest.raises(check_test_report.ReportError, match="failure"):
        check_test_report.check_report(_report(tmp_path, failures=1))


def test_rejects_collection_error(tmp_path):
    with pytest.raises(check_test_report.ReportError, match="error"):
        check_test_report.check_report(_report(tmp_path, errors=1))


def test_rejects_skip_or_xfail(tmp_path):
    with pytest.raises(check_test_report.ReportError, match="skipped"):
        check_test_report.check_report(_report(tmp_path, skipped=1))


def test_rejects_count_below_floor(tmp_path):
    with pytest.raises(check_test_report.ReportError, match="below floor"):
        check_test_report.check_report(_report(tmp_path, tests=19))


def test_rejects_summary_test_count_mismatch(tmp_path):
    with pytest.raises(check_test_report.ReportError, match="inconsistent"):
        check_test_report.check_report(_report(tmp_path, declared_tests=25))


def test_rejects_missing_counter(tmp_path):
    path = _report(tmp_path)
    tree = ET.parse(path)
    del tree.getroot()[0].attrib["errors"]
    tree.write(path, encoding="utf-8")
    with pytest.raises(check_test_report.ReportError, match="missing"):
        check_test_report.check_report(path)


def test_rejects_noninteger_counter(tmp_path):
    path = _report(tmp_path)
    tree = ET.parse(path)
    tree.getroot()[0].set("tests", "twenty")
    tree.write(path, encoding="utf-8")
    with pytest.raises(check_test_report.ReportError, match="invalid"):
        check_test_report.check_report(path)


def test_rejects_negative_counter(tmp_path):
    path = _report(tmp_path)
    tree = ET.parse(path)
    tree.getroot()[0].set("failures", "-1")
    tree.write(path, encoding="utf-8")
    with pytest.raises(check_test_report.ReportError, match="negative"):
        check_test_report.check_report(path)


def test_rejects_zero_minimum_override(tmp_path):
    with pytest.raises(check_test_report.ReportError, match="at least 1"):
        check_test_report.check_report(_report(tmp_path), min_tests=0)


def test_cli_returns_success_for_valid_report(tmp_path, capsys):
    assert check_test_report.main([str(_report(tmp_path))]) == 0
    assert "ACCEPT: 20 tests" in capsys.readouterr().out


def test_cli_returns_failure_for_invalid_report(tmp_path, capsys):
    assert check_test_report.main([str(_report(tmp_path, tests=19))]) == 1
    assert "REJECT:" in capsys.readouterr().err
