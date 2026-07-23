# Verification: SEEV hardening, CI, and project site

## Verdict

PASS for all local and live acceptance criteria.

## Independent evidence

- Clean Python 3.10.18 environment: 97 focused unit/CI tests passed.
- JUnit integrity gate: 97 tests, zero failures, errors, or skips.
- Pillow gate: installed 12.3.0; both direct constraints require 12.3.0 or
  newer.
- Site: 36 site tests passed; forced-clean Sphinx build produced seven pages
  with warnings treated as errors.
- Required CI job IDs: `unit` and `site`.
- Optimized `generate_samples`: bit-for-bit parity for float64, integer, and
  float32 domains in the required unit gate.
- Independent benchmark: optimized median 36.50 microseconds; legacy median
  91.54 microseconds; ratio 0.399.
- Tracked raw benchmark: optimized median 36.96 microseconds; legacy median
  92.08 microseconds; ratio 0.401.
- Pages workflow: official configure/upload/deploy actions, job-scoped Pages
  and OIDC permissions, `github-pages` environment, and
  `site/_build/html` artifact.
- Full `origin/main...HEAD` diff inspected; `git diff --check` passed.

## Live evidence

- PR #5 passed required `unit` and `site` checks before merge.
- PR #5 merged to `main` as `463ee5c`.
- `main` requires pull requests and strict `unit`/`site` checks, enforces the
  rules for administrators, requires conversation resolution and linear
  history, and rejects force pushes and deletions.
- Pages run `30030915659` completed successfully.
- `https://hongchaozhang-hz.github.io/SEEV/` returned HTTP 200 with the expected
  SEEV landing-page content.

## Boundaries

- The licensed solver-backed research path was not run and is not a required
  pull-request gate.
- `actionlint` was unavailable. YAML parsing and workflow contract tests
  passed.
- The Pillow requirement and installed environment are remediated. GitHub
  Dependabot alert closure is asynchronous and is checked separately from the
  acceptance verdict.
