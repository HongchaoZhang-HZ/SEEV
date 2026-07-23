# Supervised execution plan v1

Status: Approved and immutable

Base: `445065d12313367df5a7fd3cfd11679b9f257d53`

Goal: harden SEEV with a focused unit/CI contract, remediate Pillow, improve one
measured hot path, publish a clear project site, and protect `main`.

## M1 — Unit surface and Pillow remediation

- Objective: create a license-free, Python 3.10+ unit surface and remove the
  vulnerable Pillow constraint.
- Dependencies: none.
- Allowed scope: dependency metadata, the minimum `EEV/EEV` modules needed by
  the focused unit surface, `tests/unit`, and `scripts/ci`.
- Acceptance gate: focused tests pass; the JUnit integrity checker passes; the
  resolved Pillow version and every direct Pillow constraint are at least
  12.3.0.
- Validation:
  - `.venv/bin/python -m pytest tests/unit -q --junitxml=artifacts/unit.xml`
  - `.venv/bin/python scripts/ci/check_test_report.py artifacts/unit.xml`
  - `.venv/bin/python scripts/ci/check_pillow.py`
  - `git diff --check`
- Evidence path: `artifacts/m1/`.
- Stop condition: stop on a required scientific-behavior change, solver-license
  dependency in the focused surface, or scope conflict.

## M2 — CI and measured optimization

- Objective: add stable CI gates and improve one observed core hot path without
  behavior drift.
- Dependencies: M1 accepted by Codex.
- Allowed scope: `.github/workflows/ci.yml`, one core implementation path,
  `tests/performance`, and directly related tests/support files.
- Acceptance gate: local equivalents of CI pass; the optimized path has
  behavior regression coverage; recorded final median is no slower than the
  recorded baseline.
- Validation:
  - `.venv/bin/python -m pytest tests/unit tests/performance -q`
  - `.venv/bin/python scripts/ci/check_test_report.py artifacts/unit.xml`
  - `git diff --check`
- Evidence path: `artifacts/m2/`.
- Stop condition: stop if performance evidence is unstable, the candidate
  requires API changes, or broader cleanup would be needed.

## M3 — Project site and Pages workflow

- Objective: build a warning-clean Sphinx site that explains SEEV and how to use
  it, with a SNNV-shaped site tree and dmux-inspired documentation presentation.
- Dependencies: M1 accepted by Codex.
- Allowed scope: `site/`, `.github/workflows/pages.yml`, site job additions to
  CI, and narrow README navigation updates.
- Acceptance gate: warning-clean Sphinx build; required factual sections are
  present; generated navigation and copy controls work without external
  services; Pages workflow targets `main`.
- Validation:
  - `.venv/bin/python -m sphinx -W --keep-going -b html site site/_build/html`
  - `.venv/bin/python -m pytest tests/site -q`
  - `git diff --check`
- Evidence path: `artifacts/m3/`.
- Stop condition: stop on a need to invent results, modify scientific claims,
  or change deployment outside GitHub Pages.

## M4 — Integration, independent verification, and publication

- Objective: integrate only accepted task diffs, verify all requirements, then
  publish through a pull request and protect `main`.
- Dependencies: M1, M2, and M3 accepted by Codex.
- Allowed scope: Codex-only review artifacts, Git commit/push/PR, GitHub Actions,
  Pages, and branch-protection settings.
- Acceptance gate: all local gates and required PR checks pass; `main` reports
  the required protection; the public Pages URL responds successfully.
- Validation: full plan audit plus live GitHub/API checks.
- Evidence path: `tasks/verification-seev-hardening-site.md` and supervisor
  `codex-reviews/`.
- Stop condition: stop on failed CI, insufficient GitHub permission, merge
  conflict, or a live deployment failure that cannot be reproduced locally.

## Plan invariants

- Claude Code cannot modify this plan, the PRD, or the task plan.
- Claude Code cannot push, merge, publish, or change GitHub settings.
- A failed implementation or test is a FIX under this plan, not a re-plan.
- Codex independently reruns each gate before recording PASS.
- The user's dirty checkout remains untouched.
