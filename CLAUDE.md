# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`rms-polymath` (import name `polymath`) is a NumPy wrapper adding masks, units, derivatives, and
3-D geometry types. Single package, `src/` layout.

## Detailed rules

`.claude/rules/*.md` hold the authoritative detailed standards (Python style, testing,
documentation, dependencies, environment) and load automatically. The `doc_*` and
`how_to` rules are scoped to the files they govern, so they load only when you touch
`README.md` or `docs/`. Process standards live in `.claude/skills/` and load on demand:
`git-workflow`, `pull-request`, `bug-report`. This file records only what you would
otherwise get wrong.

## Verifying changes

`scripts/run-all-checks.sh` is the single source of truth for which checks must pass — CI is
required to run exactly that set. Run it after any change.

- It requires an activated-able virtualenv at `./venv` (override with `VENV`). Create it with
  `./scripts/setup-venv.sh`, which is idempotent. Never install into system Python.
- Useful flags: `-c/--code`, `-d/--docs`, `-m/--markdown`, `-s/--sequential`, or a single check
  such as `--pytest` / `--ruff-check` / `--sphinx`.
- `ruff format`, `mypy`, `bandit`, and `vulture` are disabled by default in the script. Leave them
  disabled; in particular see the mypy rule below.
- Docs preview: `./scripts/read-docs.sh`.

## Python style

- **Ruff is the linter of record** for every rule it implements. `ruff check src tests` must
  pass; it runs in CI and in the check script.
- Ruff implements no rule in the `E121`-`E133` range, so continuation-line indentation is the
  one pycodestyle family it cannot gate — no amount of `preview` changes that, because the
  rules are not written. `flake8 --select=E12,E13 src tests` covers it, in CI and as
  `--flake8-cont` in the check script. That makes the `per-file-ignores` in `.flake8`
  authoritative for those codes alone; for every other code `.flake8` is still manual-use
  only. The exemptions there cover deliberate column alignment that pycodestyle cannot know
  about, so read the comment before removing one.
- **Maximum line length 90.** Test files are exempt from E501.
- Several rules are switched off deliberately in `pyproject.toml`, each with the reason beside
  it — notably `RUF005` (Qube overloads `+`, so the rule cannot tell vector addition from list
  concatenation) and `I001` (its fix collapses the column-aligned imports used throughout).
  Read the comment before re-enabling one.
- Single quotes (`[tool.ruff.format] quote-style = "single"`).
- **Never use type annotations anywhere under `src/`** — parameter and return types belong in the
  docstrings. **Annotate all test functions and methods**, including `-> None`.
- The package ships a PEP 561 `py.typed` marker, so public type information goes in `.pyi` stubs
  alongside the modules. A stub replaces its module entirely for type checkers: whatever the stub
  omits becomes invisible downstream, so a new stub must cover the module's whole public surface.
  `stubtest` enforces exactly that and runs in the check script and in CI, so adding, renaming or
  re-signing any public member means updating its stub in the same change. Most of `Qube`'s methods
  are bound on at import time from `extensions/`, and they all have to appear in `qube.pyi`.
  Signature shapes in the stubs are exact; types come from the docstrings where those state one
  and are `Any` where they do not, which is deliberate rather than an omission to fill in blindly.
- `qube.py` holds only what defines an object: the class constants, `__init__`, the construction
  path, low-level access, the properties and the cache. Everything else lives in `extensions/` and
  is bound onto `Qube` by `extensions/__init__.py`. Two rules keep that working. First,
  `polymath/__init__.py` must import `polymath.extensions` **before** any subclass module, because
  each subclass builds read-only constants such as `Scalar.ZERO` as it loads and those calls need
  the bound methods; for the same reason no module under `extensions/` may import a subclass at
  module level — reach for `Qube._SCALAR_CLASS` and friends instead. Second, a `@staticmethod` or
  `@property` can be written at module level and bound directly, but a module-level `@classmethod`
  is not a function and `stubtest` rejects it: write the plain function and wrap it at the binding
  site, as `_suitable_dtype` does.
- **Never run `mypy` on `src/`.** `[tool.mypy] strict = true` is configured but `src/` is
  deliberately unannotated, so it would produce meaningless errors. Run mypy on `tests/` only.
- Run `ruff check src tests` after changes. Do not disable the `A` (builtins) or `N` (naming) rule
  categories.
- `.flake8` deliberately ignores whitespace-alignment codes (E201, E203, E221, E241, …) — the
  codebase uses aligned assignments on purpose. Do not reformat that alignment away.
- At most 5 positional parameters; the rest keyword-only after `*`.
- No unicode smart quotes, em-dashes, or arrows inside `.py` files (they are fine in `.rst`/`.md`).
- Make the minimal change the task requires, and match the style of the surrounding file.

## Docstrings

Every module, class, function, and method needs one: PEP 257, Google style, but using
`Parameters:` — **not** `Args:`. Wrap to 90 characters. A docstring must be detailed enough that a
black-box test can be written from it alone. Never mention backwards compatibility, change
history, user requests, or issue numbers in a docstring.

## Testing

- The suite is **pytest throughout** — module-level `test_*` functions, no `unittest.TestCase`.
  Write new tests the same way: plain `assert`, fixtures, `pytest.raises(..., match=...)`,
  `@pytest.mark.parametrize`.
- Keep tests **independent**: each function reseeds `np.random` itself and defines the values
  it needs, so it passes when run alone. Don't rely on a value another test left behind.
- Prefer one behavior per test function so a failure names what broke. Where a file's setup is
  genuinely sequential, a longer function is fine — correctness before granularity.
- `pytest` addopts already apply `-n auto --cov=src/polymath --strict-markers
  --strict-config`, so every run is parallel with coverage. Tests must be order-independent.
- `markers` is an empty list plus `--strict-markers`: any unregistered `@pytest.mark.<custom>`
  fails the run. Register it in `pyproject.toml` first.
- Coverage must stay at or above 90% (`fail_under = 90`, branch coverage on). The pytest
  `--cov` target and `[tool.coverage.run] source` must name the same path, or the floor
  measures something other than what the run covered.
- `filterwarnings = ["error"]`: any warning a test triggers fails it. The suite passes
  with no exemptions, so add a narrowly-scoped `ignore::` entry only for a warning from
  third-party code you cannot fix, with a comment saying why.
- Assert one condition per `assert` (no `and`), on exact expected values; `pytest.approx` for
  floats.
- There is no `conftest.py` yet, and `tests/` is flat — it does not mirror `src/polymath/extensions/`.
- `assertAlmostEqual` was translated as `a == b or abs(a - b) <= tol`, not `pytest.approx`.
  `approx` does not understand `Qube` operands, and the `==` arm reproduces unittest's
  short-circuit, which is what let masked values compare equal.

## Documentation

Sphinx builds are warning-as-error (`-W`) in CI, in the check script, and in `read-docs.sh`, so any
new warning breaks the build. Narrative docs are `.rst`; Markdown is only for README/CONTRIBUTING
via MyST. Every API symbol named in prose must use a Sphinx role (`:class:`, `:meth:`, `:func:`,
`:mod:`, `:attr:`, `:data:`) — a bare CamelCase name or inline literal is a violation. American
spelling, one space after a sentence-ending period, no time-anchored words ("new", "legacy", "now").

The check script and CI scan the same Markdown set — `docs/`, `.claude/`, `README.md` and
`CONTRIBUTING.md`. Keep the two in step if either changes.

## Repo etiquette

- Commit subjects: plain capitalized imperative sentence, no type prefix, no trailing period
  (e.g. `Increase test coverage, improve docstrings, minor bug fixes (#13)`). PRs are squash-merged
  onto `main`, which appends the `(#N)`.
- Branch names follow `<initials>_<YYMMDD>_<topic>`, e.g. `rf_251204_mixins`.
- Never commit `build/`, `.coverage`, `.pytest_cache/`, or `src/rms_polymath.egg-info/`.
- Dependencies go in `pyproject.toml` only; `requirements.txt` contains just `-e .`. Use minimum
  version constraints, never `==` pins.
- Versions come from `setuptools_scm`; never hand-edit `src/polymath/_version.py`.
