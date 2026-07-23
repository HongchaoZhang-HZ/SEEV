#!/usr/bin/env python3
"""Strict JUnit test-report integrity checker.

A passing pytest run is not sufficient evidence on its own: a report can be
missing, truncated, contain collection errors, silently skip tests, or cover
far fewer tests than expected. This checker rejects all of those conditions and
only accepts a report that:

* exists and is well-formed XML,
* records at least a nonzero floor of executed tests,
* has zero failures, zero errors (including collection errors), and
* has zero skipped/xfailed tests (no silent relaxation).

Usage:
    python check_test_report.py <junit.xml> [--min-tests N]

Exit status is ``0`` when the report is accepted and ``1`` when it is rejected.
"""

from __future__ import annotations

import argparse
import os
import sys
from xml.etree import ElementTree as ET

# Recorded nonzero floor for the focused M1 surface (unit + ci suites are well
# above this). Overridable for reuse by later milestones.
DEFAULT_MIN_TESTS = 20


class ReportError(Exception):
    """Raised when a JUnit report fails an integrity requirement."""


_COUNTERS = ("tests", "failures", "errors", "skipped")


def _counter(element: ET.Element, key: str) -> int:
    raw = element.get(key)
    if raw is None:
        raise ReportError(f"<testsuite> is missing the {key!r} counter")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ReportError(f"<testsuite> has invalid {key!r} counter: {raw!r}") from exc
    if value < 0:
        raise ReportError(f"<testsuite> has negative {key!r} counter: {value}")
    return value


def parse_report(path: str) -> dict:
    """Parse a JUnit XML file into aggregate counters.

    Raises ``ReportError`` if the file is missing or not well-formed XML.
    """
    if not os.path.isfile(path):
        raise ReportError(f"report not found: {path}")
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError) as exc:
        raise ReportError(f"malformed JUnit XML: {exc}") from exc

    root = tree.getroot()
    if root.tag == "testsuite":
        summary_suites = [root]
    elif root.tag == "testsuites":
        summary_suites = [child for child in root if child.tag == "testsuite"]
    else:
        raise ReportError(f"unexpected JUnit root element: <{root.tag}>")
    if not summary_suites:
        raise ReportError("no top-level <testsuite> element found")

    totals = {key: 0 for key in _COUNTERS}
    for suite in summary_suites:
        for key in _COUNTERS:
            totals[key] += _counter(suite, key)

    testcases = list(root.iter("testcase"))
    actual = {
        "tests": len(testcases),
        "failures": sum(len(case.findall("failure")) for case in testcases),
        "errors": sum(len(case.findall("error")) for case in testcases),
        "skipped": sum(len(case.findall("skipped")) for case in testcases),
    }
    for key in _COUNTERS:
        if totals[key] != actual[key]:
            raise ReportError(
                f"inconsistent {key!r} counter: summary records {totals[key]}, "
                f"testcases record {actual[key]}"
            )
    return totals


def check_report(path: str, min_tests: int = DEFAULT_MIN_TESTS) -> dict:
    """Validate a JUnit report; raise ``ReportError`` if it is unacceptable."""
    if min_tests < 1:
        raise ReportError("minimum test count must be at least 1")
    totals = parse_report(path)
    if totals["errors"] > 0:
        raise ReportError(f"report has {totals['errors']} error(s) / collection error(s)")
    if totals["failures"] > 0:
        raise ReportError(f"report has {totals['failures']} failure(s)")
    if totals["skipped"] > 0:
        raise ReportError(f"report has {totals['skipped']} skipped/xfail test(s)")
    if totals["tests"] < min_tests:
        raise ReportError(
            f"report records {totals['tests']} test(s), below floor of {min_tests}"
        )
    return totals


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Strict JUnit report checker")
    parser.add_argument("report", help="path to the JUnit XML file")
    parser.add_argument(
        "--min-tests",
        type=int,
        default=DEFAULT_MIN_TESTS,
        help=f"minimum executed test count (default: {DEFAULT_MIN_TESTS})",
    )
    args = parser.parse_args(argv)
    try:
        totals = check_report(args.report, args.min_tests)
    except ReportError as exc:
        print(f"REJECT: {exc}", file=sys.stderr)
        return 1
    print(
        "ACCEPT: {tests} tests, {failures} failures, {errors} errors, "
        "{skipped} skipped".format(**totals)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
