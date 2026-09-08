==================
Coding Conventions
==================

The authoritative rules live in the repository, in ``CLAUDE.md`` and under
``.claude/rules/``. Those files are written to guide AI-assisted development, but they
apply to every contributor, and the checks enforce most of them. This chapter summarizes
the rules a developer is most likely to trip over.

Python Style
============

* Ruff is the linter of record for every rule it implements, with the rule set in
  ``pyproject.toml``. Each disabled rule has its reason beside it; read the reason before
  re-enabling one. Do not disable the ``A`` (builtins) or ``N`` (naming) categories.
* Ruff has no rule for continuation-line indentation, so flake8 checks codes E12x and
  E13x, reading the per-file exemptions in ``.flake8``. Those exemptions cover deliberate
  column alignment; read the comment before removing one.
* The maximum line length is 90 characters. Test files are exempt.
* Use single quotes.
* The code base aligns assignments and imports in columns on purpose, and the
  whitespace rules that would object are switched off. Do not reformat that alignment
  away, and match the style of the surrounding file.
* At most five positional parameters; the rest are keyword-only after ``*``.
* No unicode smart quotes, em-dashes, or arrows inside ``.py`` files.
* No type annotations under ``src/``, except the return annotation of a property. See
  :doc:`dev_guide_typing`.
* Make the minimal change the task requires.

Docstrings
==========

Every module, class, function, and method has a docstring in Google style, using
``Parameters:`` rather than ``Args:``, with ``Returns:`` and ``Raises:`` where they apply,
wrapped to 90 characters. A docstring must be detailed enough that a black-box test can
be written from it alone. It describes observable behavior only, never implementation
details, change history, backward compatibility, or an issue number.

Tests
=====

* The suite is pytest throughout: module-level ``test_*`` functions, plain ``assert``,
  fixtures, ``pytest.raises`` with ``match=``, and ``pytest.mark.parametrize``. No
  ``unittest.TestCase``.
* Every test function and method is annotated, including ``-> None``.
* Each test is independent. It seeds NumPy's random generator itself and defines every
  value it needs, so that it passes alone and in any order under parallel execution.
* One behavior per test function where practical, so a failure names what broke. One
  condition per ``assert``, on an exact expected value; ``pytest.approx`` for floats.
* Any warning raised during a test fails it. Add an ``ignore::`` entry only for a warning
  from third-party code, with a comment saying why.
* Register any custom marker in ``pyproject.toml`` before using it.
* Coverage stays at or above 90 percent over the whole suite.

Documentation
=============

* Narrative documentation is reStructuredText under ``docs/``; Markdown is only for the
  files that must also render on GitHub.
* Builds are warning-as-error and nitpicky everywhere. Every API symbol named in prose
  uses a Sphinx role; a bare CamelCase name or an inline literal is a violation, and an
  inline literal is reserved for file paths, keys, and shell tokens.
* Cross-references to third-party objects use the spelling their documentation exports,
  such as ``numpy.ndarray`` rather than ``np.ndarray``. Never add a nitpick exemption
  for a symbol this project owns.
* American spelling, one space after a sentence-ending period, and no time-anchored
  words such as "new", "legacy", or "now".
* Any code change updates the affected docstrings, guide chapters, and README in the
  same change.

Repository Etiquette
====================

* Branch names follow ``<initials>_<YYMMDD>_<topic>``.
* Commit subjects are plain capitalized imperative sentences with no type prefix and no
  trailing period. Pull requests are squash-merged, which appends the pull request
  number.
* Dependencies go in ``pyproject.toml`` only, with minimum version constraints and never
  exact pins. ``requirements.txt`` contains just ``-e .``.
* Never commit ``build/``, ``.coverage``, ``.pytest_cache/``, ``htmlcov/``, or
  ``src/rms_polymath.egg-info/``, and never hand-edit ``src/polymath/_version.py``.
