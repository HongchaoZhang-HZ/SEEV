# Task Plan: SEEV hardening, CI, and project site

## Status

Local verification passed; publication pending

## Inputs

- PRD: `tasks/prd-seev-hardening-site.md`
- Supervisor plan: `tasks/supervisor-plan-v1.md`
- Approved scope: code cleanup, focused unit tests and CI, Pillow remediation,
  one measured optimization, a SNNV-shaped/dmux-inspired site, live Pages, and
  `main` protection.

## Relevant Files

### Planned

- `requirements.txt` — remove the vulnerable Pillow constraint and document the
  supported baseline.
- `requirements-ci.txt` — focused, license-free CI environment.
- `EEV/EEV/` — package/import cleanup and one measured hot-path improvement.
- `tests/` — focused unit and performance tests.
- `scripts/ci/` — JUnit and dependency integrity checks.
- `.github/workflows/ci.yml` — pull-request unit and site build gates.
- `.github/workflows/pages.yml` — GitHub Pages deployment from `main`.
- `site/` — Sphinx project site and usage documentation.
- `README.md` — accurate entry points and site link after publication.

### Actually Changed

- `tasks/prd-seev-hardening-site.md` — approved requirements.
- `tasks/tasks-seev-hardening-site.md` — approved workflow task plan.
- `tasks/supervisor-plan-v1.md` — immutable Claude supervision plan.
- `EEV/EEV/`, `tests/`, and `scripts/ci/` — focused imports, behavior tests,
  strict report/dependency gates, and measured sample-generation optimization.
- `requirements.txt` and `requirements-ci.txt` — Pillow 12.3.0 minimum and
  Python 3.10+ focused environment.
- `.github/workflows/` — stable `unit` and `site` pull-request gates plus Pages
  deployment.
- `site/`, `requirements-docs.txt`, and `README.md` — warning-clean project
  site, usage documentation, and public entry point.
- `artifacts/m2/benchmark.json` and `artifacts/m3/VALIDATION.md` — raw benchmark
  and site validation evidence.
- `tasks/verification-seev-hardening-site.md` — independent verifier verdict.

## Validation Strategy

- Focused: `.venv/bin/python -m pytest tests/unit -q --junitxml=artifacts/unit.xml`
- Report gate: `.venv/bin/python scripts/ci/check_test_report.py artifacts/unit.xml`
- Dependency gate: `.venv/bin/python scripts/ci/check_pillow.py`
- Site: `.venv/bin/python -m sphinx -W --keep-going -b html site site/_build/html`
- Diff: `GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null git diff --check`
- Live: GitHub Actions, branch-protection API, Pages deployment status, and
  public URL response.

## Tasks

- [x] 1.0 Establish the maintained unit-test surface and Pillow policy
  - [x] 1.1 Refactor only the core package imports needed by focused tests.
    - Dependencies: none
    - Scope: `EEV/EEV`, packaging metadata
    - Owner: Claude executor; Codex reviewer
    - Acceptance: focused modules import without licensed solvers.
    - Validate: `.venv/bin/python -m pytest tests/unit -q`
    - Evidence: focused solver-free imports pass in the 97-test CI gate.
  - [x] 1.2 Add behavior-focused unit tests and a strict JUnit integrity gate.
    - Dependencies: 1.1
    - Scope: `tests/unit`, `scripts/ci`
    - Owner: Claude executor; Codex reviewer
    - Acceptance: tests pass; report has no failures, errors, unexpected skips,
      and meets its recorded count floor.
    - Validate: `.venv/bin/python -m pytest tests/unit -q --junitxml=artifacts/unit.xml && .venv/bin/python scripts/ci/check_test_report.py artifacts/unit.xml`
    - Evidence: JUnit accepted 97 tests with zero failures, errors, or skips.
  - [x] 1.3 Replace Pillow 9.5.0 with a Python 3.10+ compatible safe constraint.
    - Dependencies: none
    - Scope: dependency files and dependency check
    - Owner: Claude executor; Codex reviewer
    - Acceptance: resolved Pillow is at least 12.3.0 and no direct requirement
      permits an older release.
    - Validate: `.venv/bin/python scripts/ci/check_pillow.py`
    - Evidence: installed Pillow 12.3.0; both direct constraints accepted.

- [x] 2.0 Add CI and measured optimization
  - [x] 2.1 Add pull-request unit/report and site-build jobs.
    - Dependencies: 1.0
    - Scope: `.github/workflows/ci.yml`, CI support files
    - Owner: Claude executor; Codex reviewer
    - Acceptance: workflow syntax is valid and local equivalents pass.
    - Validate: local unit/report/dependency/site commands plus GitHub run
    - Evidence: local YAML/contract validation passed; live run pending.
  - [x] 2.2 Optimize one profiled core hot path.
    - Dependencies: 1.2
    - Scope: one core implementation path and its tests/benchmark
    - Owner: Claude executor; Codex reviewer
    - Acceptance: behavior is unchanged and final benchmark median is no slower
      than the recorded baseline.
    - Validate: `.venv/bin/python -m pytest tests/unit tests/performance -q`
    - Evidence: tracked median ratio 0.401; float64, integer, and float32
      behavior is bit-for-bit identical in the required unit gate.

- [x] 3.0 Build the public project site
  - [x] 3.1 Create the Sphinx site, responsive visual system, and usage content.
    - Dependencies: 1.0
    - Scope: `site/`, factual links and commands from repository sources
    - Owner: Claude executor; Codex reviewer
    - Acceptance: required pages/content are present, commands are copyable, and
      the warning-clean site build passes.
    - Validate: `.venv/bin/python -m sphinx -W --keep-going -b html site site/_build/html`
    - Evidence: 36 site tests; seven warning-clean pages; desktop/mobile browser
      checks found one H1 and no horizontal overflow.
  - [x] 3.2 Add Pages deployment and README entry points.
    - Dependencies: 3.1
    - Scope: `.github/workflows/pages.yml`, `README.md`
    - Owner: Claude executor; Codex reviewer
    - Acceptance: Pages workflow builds the same `site/_build/html` output.
    - Validate: local site build plus GitHub Pages run
    - Evidence: local Pages workflow contract passed; live deployment pending.

- [ ] 4.0 Independently verify and publish
  - [x] 4.1 Audit the final diff against every PRD criterion.
    - Dependencies: 2.0, 3.0
    - Scope: read-only verification and `tasks/verification-seev-hardening-site.md`
    - Owner: Codex and independent verifier
    - Acceptance: required gates pass with no unreported scope deviation.
    - Validate: all focused and broad commands in this plan
    - Evidence: independent verifier PASS at `d00d03b`; see
      `tasks/verification-seev-hardening-site.md`.
  - [ ] 4.2 Push the feature branch, open a pull request, and require passing CI.
    - Dependencies: 4.1
    - Scope: GitHub branch and pull request
    - Owner: Codex
    - Acceptance: required PR checks pass.
    - Validate: GitHub API and Actions state
    - Evidence: pending
  - [ ] 4.3 Protect `main`, merge through the pull request, enable Pages, and
        verify the live site.
    - Dependencies: 4.2
    - Scope: GitHub repository settings and merge
    - Owner: Codex
    - Acceptance: branch protection and public Pages URL satisfy AC-6 and AC-7.
    - Validate: GitHub API plus public URL request
    - Evidence: pending

## Deviations and Decisions

- Work starts from `origin/main`, not the dirty local checkout, because the
  checkout is one commit behind and contains 231 uncommitted deletions.
- The mandatory CI surface excludes licensed and unavailable research solvers;
  those paths remain documented integration workflows.

## Blockers

- None at plan approval.
