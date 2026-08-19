# Agent Guidelines

These instructions apply to the entire repository.

## Project conventions

- Use Python 3.12 and manage dependencies exclusively with `uv`.
- Treat `pyproject.toml` as the source of truth for project and tool
  configuration. Do not create duplicate tool configuration files unless a tool
  requires one.
- Keep reusable application and ML logic under `src/cycle_forecast/`. Keep
  notebooks exploratory; do not make production code depend on notebooks.
- Never commit personal health data, credentials, trained models, experiment
  artifacts, or identifying notebook output.

## Git and pull-request workflow

- Treat `master` as a protected integration branch. Do not make feature, fix,
  documentation, or maintenance commits directly on `master`.
- Before starting a coherent change, inspect the branch and working tree. Start
  from an up-to-date, clean `master`, then create a focused branch such as
  `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, or `chore/<topic>`.
- Never discard, overwrite, relocate, or include unrelated working-tree changes.
  If pre-existing changes make creating a clean branch unsafe, stop and ask for
  direction.
- Keep commits focused and use descriptive conventional-commit messages. Stage
  only files that belong to the current change.
- Push the branch and open a pull request instead of pushing changes directly to
  `master`. The pull request should summarize the change, its motivation, and
  the verification performed.
- Let the complete GitHub Actions workflow run on the pull request. Do not merge
  until every required check passes; fix failures on the same branch and rerun
  the checks.
- Merge through the pull request after checks pass and review is complete, then
  delete the merged branch when appropriate.
- Pushing branches, opening or editing pull requests, and merging or closing
  pull requests change remote state. Perform those actions only when the user
  has requested or authorized them.

## Python standards

- Fully annotate every function and method, including parameters and return
  types. Do not introduce untyped functions to silence the type checker.
- Write a NumPy-style docstring for every function, method, class, and module.
  Include `Parameters`, `Returns`, `Raises`, and `Notes` sections when they are
  applicable; omit empty sections.
- Keep Pyright strict mode passing. Prefer precise types and type narrowing over
  `Any`, casts, or ignore comments. If suppression is unavoidable, make it as
  narrow as possible and explain why.
- Use Ruff for formatting, import sorting, linting, and docstring enforcement.
  Do not manually format code contrary to the Ruff configuration.
- Share feature transformations between training and inference code. Never
  duplicate feature logic in notebooks or serving code.

## Tests

- Put tests under `tests/`, mirroring the corresponding path beneath
  `src/cycle_forecast/`.
- Name test modules `test_<module>.py` and test functions `test_<behavior>`.
- Add or update tests for every behavior change and bug fix. Focus especially on
  data validation, feature correctness, temporal leakage, and boundary cases.
- Keep tests deterministic. Do not make unit tests depend on private data,
  network access, wall-clock time, or execution order.
- Use synthetic, non-identifying fixtures for examples and tests.
- During iteration, run the most relevant tests. Before handing work back, run
  Ruff, Pyright, and the complete pytest suite, even though pre-commit uses
  pytest-testmon to select affected tests.

## Required verification

Run these commands before considering a change complete:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
uv build
```
