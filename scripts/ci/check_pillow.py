#!/usr/bin/env python3
"""Strict Pillow supply-chain policy checker.

CVE-affected Pillow releases must not be reachable from the maintained
environment. This checker enforces a single policy in two places:

* the *installed* Pillow must be at least :data:`MIN_PILLOW`, and
* every *direct* Pillow constraint declared in the maintained requirement files
  must forbid any resolution below :data:`MIN_PILLOW` (no exact pin below the
  floor, and no unbounded-below or upper-bound-only specifier).

Usage:
    python check_pillow.py [--root DIR]

Exit status is ``0`` when every check passes and ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from packaging.version import InvalidVersion, Version

MIN_PILLOW = Version("12.3.0")

# Requirement files on the maintained path that may declare Pillow directly.
CONSTRAINT_FILES = ("requirements.txt", "requirements-ci.txt")

_CLAUSE_RE = re.compile(r"(===|==|>=|<=|~=|!=|>|<)\s*([0-9][A-Za-z0-9_.*+!-]*)")
_PILLOW_LINE_RE = re.compile(
    r"^\s*pillow(?=\s*(?:\[|@|===|==|>=|<=|~=|!=|>|<|;|$))",
    re.IGNORECASE,
)


class PillowError(Exception):
    """Raised when the Pillow policy is violated."""


def parse_version(text: str) -> Version:
    """Parse a PEP 440 version, normalizing a final wildcard to its floor."""
    value = str(text).strip()
    if value.endswith(".*"):
        value = value[:-2] + ".0"
    try:
        return Version(value)
    except InvalidVersion as exc:
        raise PillowError(f"invalid Pillow version: {text!r}") from exc


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0]


def clauses_from_line(line: str):
    """Return ``[(operator, version_tuple), ...]`` for a requirement line."""
    return [
        (op, parse_version(ver))
        for op, ver in _CLAUSE_RE.findall(_strip_comment(line))
    ]


def is_pillow_line(line: str) -> bool:
    return bool(_PILLOW_LINE_RE.match(_strip_comment(line)))


def line_is_compliant(line: str) -> bool:
    """True iff a Pillow requirement line forbids versions below the floor."""
    try:
        clauses = clauses_from_line(line)
    except PillowError:
        return False
    lower_bounds = [ver for op, ver in clauses if op in ("==", "===", ">=", ">", "~=")]
    if not lower_bounds:
        # No lower bound (e.g. only "<=" / "<" or a bare name) permits old versions.
        return False
    # Every lower-bound clause must independently sit at or above the floor.
    return all(ver >= MIN_PILLOW for ver in lower_bounds)


def installed_version() -> Version:
    try:
        import PIL
    except ImportError as exc:  # pragma: no cover - Pillow is a hard requirement
        raise PillowError("Pillow is not installed") from exc
    return parse_version(PIL.__version__)


def check_installed() -> Version:
    version = installed_version()
    if version < MIN_PILLOW:
        raise PillowError(
            "installed Pillow {} is below required {}".format(
                version, MIN_PILLOW
            )
        )
    return version


def check_file(path: str) -> int:
    """Validate every Pillow constraint in one requirement file.

    Returns the number of Pillow constraints checked. Raises ``PillowError`` on
    the first non-compliant constraint.
    """
    if not os.path.isfile(path):
        raise PillowError(f"requirement file not found: {path}")
    found = 0
    with open(path, encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            if not is_pillow_line(raw):
                continue
            found += 1
            if not line_is_compliant(raw):
                raise PillowError(
                    f"{os.path.basename(path)}:{lineno}: non-compliant Pillow "
                    f"constraint: {raw.strip()!r}"
                )
    return found


def check_repo(root: str, files=CONSTRAINT_FILES) -> dict:
    """Check the installed version and every constraint file under ``root``."""
    result = {"installed": check_installed(), "constraints": 0}
    for name in files:
        result["constraints"] += check_file(os.path.join(root, name))
    if result["constraints"] == 0:
        raise PillowError("no direct Pillow constraint found on the maintained path")
    return result


def _default_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Strict Pillow policy checker")
    parser.add_argument("--root", default=_default_root(), help="repository root")
    args = parser.parse_args(argv)
    try:
        result = check_repo(args.root)
    except PillowError as exc:
        print(f"REJECT: {exc}", file=sys.stderr)
        return 1
    print(
        "ACCEPT: installed Pillow {}, {} direct constraint(s) checked".format(
            result["installed"], result["constraints"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
