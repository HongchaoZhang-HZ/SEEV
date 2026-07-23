# M3-T01 validation notes

Validation run from the isolated M3 worktree on 2026-07-23.

## Acceptance commands

```text
$ .venv/bin/python -m sphinx -W --keep-going -b html site site/_build/html
build succeeded.
The HTML pages are in site/_build/html.

$ .venv/bin/python -m pytest tests/site tests/unit tests/ci -q --junitxml=artifacts/m3/tests.xml
129 passed in 3.66s

$ .venv/bin/python scripts/ci/check_test_report.py artifacts/m3/tests.xml
ACCEPT: 129 tests, 0 failures, 0 errors, 0 skipped

$ .venv/bin/python scripts/ci/check_pillow.py
ACCEPT: installed Pillow 12.3.0, 2 direct constraint(s) checked

$ git diff --check
[exit 0; no output]
```

## Contract evidence

- Both `.github/workflows/ci.yml` and `.github/workflows/pages.yml` parse as
  YAML.
- CI keeps the stable `unit` job and adds the stable `site` job. The workflow
  runs on pull requests and pushes to `main`; the site job uses Python 3.10,
  read-only contents permission, a timeout, `requirements-docs.txt`, and the
  warnings-as-errors Sphinx command.
- Pages runs on pushes to `main` and manual dispatch. Its separate `site` and
  `deploy` jobs use the official checkout, upload, configure, and deploy
  actions. The configure/deploy job has `pages: write`, `id-token: write`, the
  `github-pages` environment, and uploads only `site/_build/html`.
- The built landing page contains exactly one `<h1>`. The visible hero title
  remains `Synthesis with Efficient Exact Verification for ReLU Neural Barrier
  Functions`.
- The landing page links the paper and source repository. `README.md` links the
  public documentation at `https://hongchaozhang-hz.github.io/SEEV/`.
- The site contract tests cover all required pages, exact usage commands,
  maintained-vs-research setup, local assets, copy controls, responsive and
  reduced-motion CSS, workflow jobs, and Pages deployment permissions.
