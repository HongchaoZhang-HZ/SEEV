# PRD: SEEV hardening, CI, and project site

## Status

Approved

## Problem

The current remote `main` branch has no CI or branch protection, its test suite
does not collect from the repository root, and its dependency file pins
Pillow 9.5.0 despite active GitHub security alerts requiring Pillow 12.3.0 or
newer. The repository also lacks a maintained project site and a verified,
copyable usage path.

## Goal

Ship one reviewed change set that establishes a passing unit-test and site-build
gate, removes the vulnerable Pillow constraint, improves one measured core hot
path without changing its behavior, publishes project documentation, and
protects `main` with the passing CI gates.

## Users and Use Cases

- SEEV maintainers: change core verification utilities with regression evidence.
- Researchers: install the supported unit-test environment and reproduce the
  documented entry points.
- Readers: understand the SEEV method and find setup, verification, training,
  paper, and citation information from a public site.

## Functional Requirements

1. FR-1: The repository must provide a documented Python 3.10+ environment whose
   resolved Pillow version is at least 12.3.0.
2. FR-2: A focused SEEV unit suite must run without licensed solvers or external
   services and must fail on collection errors, failures, unexpected skips, an
   absent JUnit report, or a test count below the recorded floor.
3. FR-3: GitHub Actions must run the focused unit gate and build the site for
   pull requests and changes to `main`.
4. FR-4: At least one measured core hot path must be simplified or accelerated
   with behavior-preserving regression tests and before/after evidence.
5. FR-5: The site must include a project overview, method explanation, getting
   started steps, usage paths, limitations, paper link, and citation.
6. FR-6: The site must use the SNNV Sphinx/Pages structure while adopting the
   compact dark documentation layout, sticky navigation, restrained motion, and
   copyable command presentation observed on dmux.ai.
7. FR-7: `main` must require pull requests and the passing unit/site checks,
   reject force pushes and deletions, and require resolution of review
   conversations.
8. FR-8: The published site and protection settings must be verified from live
   GitHub state.

## Acceptance Criteria

- AC-1: A clean Python 3.10+ environment installs the focused test dependencies,
  and `PIL.__version__` is at least 12.3.0.
- AC-2: The focused CI test command exits 0 and its integrity checker accepts the
  generated JUnit report.
- AC-3: The site build exits 0 and contains the overview and getting-started
  pages in the built output.
- AC-4: The optimization benchmark records raw before/after results, the final
  median is no slower than the baseline, and behavior tests pass.
- AC-5: A pull request from the implementation branch has passing required
  checks before merge.
- AC-6: GitHub reports `main` as protected with the required check contexts.
- AC-7: The public site URL returns successfully after the merged Pages
  workflow completes.

## Non-Goals

- Reproducing paper-scale experiments in CI.
- Making Gurobi, dReal, Drake, or GPU workflows mandatory for pull requests.
- Rewriting the adapted `neural_clbf` research stack or changing published
  scientific claims.
- Removing model artifacts or historical experiment inputs without a separate
  provenance review.
- Resolving security alerts unrelated to Pillow unless a change is required to
  make the focused environment installable.

## Constraints

- Base all work on remote `main` commit
  `445065d12313367df5a7fd3cfd11679b9f257d53`.
- Preserve the user's dirty checkout; all implementation occurs in isolated
  worktrees.
- Claude Code may edit only its supervisor worktree and may not push, merge,
  publish, change GitHub settings, or revise the plan.
- Codex independently reviews every Claude task, reruns its gates, and owns all
  GitHub mutations.
- Do not invent benchmark results, installation success, or scientific claims.

## Design and Technical Notes

- The current root test collection fails before running tests because package
  imports and optional research dependencies are mixed into the unit surface.
- GitHub currently reports twelve open Pillow alerts with first patched version
  12.3.0.
- Pillow 12 drops Python 3.9 support, so the maintained CI path cannot retain
  the README's Python 3.9-only claim.
- SNNV uses a `site/` Sphinx tree, a dedicated Pages workflow, and a separate
  site build gate; SEEV will reuse that deployment shape without copying SNNV
  content.

## Assumptions

- Python 3.10+ is acceptable for the maintained unit and documentation path.
- The legacy full research environment remains reproducible separately and is
  not a required CI gate.
- GitHub repository administration permissions are sufficient to enable Pages
  and branch protection.

## Open Questions

- None. The user explicitly authorized end-to-end execution for this session.
