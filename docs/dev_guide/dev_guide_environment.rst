=======================
Development Environment
=======================

Getting a Working Checkout
==========================

Clone the repository and run the bootstrap script, which creates a virtual environment
at ``./venv`` and installs the package in editable mode with the ``dev`` extra. The
``dev`` extra includes the ``docs`` extra, so one command installs everything the checks
need. The script refuses an interpreter older than Python 3.11 and is safe to rerun.

.. code-block:: sh

   git clone https://github.com/SETI/rms-polymath.git
   cd rms-polymath
   ./scripts/setup-venv.sh
   source venv/bin/activate

Pass ``--python`` to choose an interpreter and ``--recreate`` to rebuild the environment
from scratch:

.. code-block:: sh

   ./scripts/setup-venv.sh --python python3.13 --recreate

Never install into the system Python. If you prefer to manage the environment yourself,
the equivalent of the script is:

.. code-block:: sh

   python3 -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"

Environment Variables
=====================

The package itself reads no environment variables. The scripts read these:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Meaning
   * - ``VENV`` or ``VENV_PATH``
     - The virtual environment the scripts activate. Default: ``./venv``. The
       ``Makefile`` in ``docs/`` honors ``VENV`` as well, so that it finds the Sphinx
       installed there rather than one on the search path.
   * - ``ENABLE_<CHECK>``
     - Per-check switches read by the check script, such as ``ENABLE_MYPY=true``. The
       defaults define the set of checks the repository has opted into; see
       `Running the Checks`_.
   * - ``CLEANUP_GRACE_PERIOD``
     - Seconds the check script waits for a check to stop after an interrupt before
       killing it. Default: 5.

Smoke Test
==========

The package has no command-line entry points. Confirm that the editable install works by
importing it and performing an operation that exercises the extension binding:

.. code-block:: sh

   python -c "import polymath; print(polymath.__version__); print(polymath.Scalar([1., 2.]) * 2)"

The version is derived from the git history by ``setuptools_scm`` and written to
``src/polymath/_version.py`` at install time, so a checkout that is not a git repository
reports ``Version unspecified``.

Running the Tests
=================

The suite is pytest throughout. The options in ``pyproject.toml`` apply to every
invocation: tests run in parallel with ``pytest-xdist``, coverage is collected for
``src/polymath``, unregistered markers and misspelled options are errors, and every
warning raised during a test fails it.

.. code-block:: sh

   pytest                                 # the whole suite, in parallel, with coverage
   pytest tests/test_pair_as_pair.py      # one file
   pytest -k swapxy                       # tests whose names match
   pytest -n 0 tests/test_qube_shrink.py  # serially, which is easier to debug
   coverage report -m                     # missing lines, after a run
   pytest --cov-report=html               # writes htmlcov/index.html

Coverage must stay at or above 90 percent, measured over the whole suite with branch
coverage on; the run fails below that. There are no slow or environment-dependent tiers
and no registered markers, so a bare ``pytest`` runs everything. Tests must be independent
and order-agnostic, because they run in parallel: each function seeds NumPy's random
generator itself and defines every value it uses.

The check script runs pytest with ``--dist loadscope``, which keeps each test module on
one worker. Use the same flag when reproducing a failure that the script reports.

Running the Checks
==================

``scripts/run-all-checks.sh`` is the single source of truth for which checks must pass.
CI runs exactly that set, no more and no less, so passing the script locally means
passing CI. Run it after every change.

.. code-block:: sh

   ./scripts/run-all-checks.sh            # everything, in parallel
   ./scripts/run-all-checks.sh -s         # everything, sequentially, easier to read
   ./scripts/run-all-checks.sh -c         # code checks only
   ./scripts/run-all-checks.sh -d         # Sphinx and Markdown only
   ./scripts/run-all-checks.sh --pytest   # one check; combine flags as needed

The checks it enables by default are:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Check
     - Flag
     - What it enforces
   * - ruff
     - ``--ruff-check``
     - The linter of record, for every rule it implements. The rule set and the
       deliberate exemptions are in ``pyproject.toml``.
   * - flake8
     - ``--flake8-cont``
     - Continuation-line indentation only (codes E12x and E13x), which ruff does not
       implement. The per-file exemptions in ``.flake8`` are authoritative for these
       codes alone.
   * - pytest
     - ``--pytest``
     - The test suite and the coverage floor.
   * - pyroma
     - ``--pyroma``
     - Packaging metadata completeness.
   * - stubtest
     - ``--stubtest``
     - The two stubs, ``__init__.pyi`` and ``typedefs.pyi``, match the runtime API. See
       :doc:`dev_guide_typing`.
   * - Sphinx
     - ``--sphinx``
     - The documentation builds with warnings as errors and with nitpicky
       cross-reference checking.
   * - PyMarkdown
     - ``--pymarkdown``
     - Markdown style for ``docs/``, ``.claude/``, ``README.md``, and ``CONTRIBUTING.md``.

Four more checks are wired in but disabled by default: ``ruff format --check``, mypy,
bandit, and vulture. Leave them disabled. In particular, never run mypy on ``src/``: the
modules there are deliberately unannotated, so it would report meaningless errors. When
the ``--mypy`` check is enabled it runs against ``tests/`` only, which are fully
annotated. Run it by hand the same way:

.. code-block:: sh

   MYPYPATH=src mypy tests

Building the Documentation
==========================

The documentation builds with ``-W``, so any warning is an error, and ``docs/conf.py``
sets nitpicky mode, so a cross-reference with no target is an error too. Both apply in
the check script, in CI, and on ReadTheDocs. Sphinx 9 or later is required, because
earlier versions cannot resolve the references to the aliases in
:mod:`polymath.typedefs`.

.. code-block:: sh

   ./scripts/run-all-checks.sh --sphinx   # build only
   ./scripts/read-docs.sh                 # build, then open in a browser

Continuous Integration
======================

Four GitHub Actions workflows live in ``.github/workflows/``.

* ``run-tests.yml`` runs on every pull request against ``main``, on every push to
  ``main``, weekly, and on demand. Its lint job runs ruff, flake8, pyroma, stubtest,
  Sphinx, and PyMarkdown on Python 3.13, which is the check script's default set minus
  pytest. Its test job runs pytest with coverage on Ubuntu, macOS, and Windows for each
  of Python 3.11, 3.12, and 3.13, and uploads coverage to Codecov from one cell of the
  matrix.
* ``audit.yml`` runs ``pip-audit`` weekly and on demand. It is deliberately not part of
  the pull request gate, because a vulnerability advisory can appear without any change
  to the repository.
* ``publish_to_pypi.yml`` builds and validates the distribution and uploads it to PyPI
  when a GitHub Release is published.
* ``publish_to_test_pypi.yml`` does the same for Test PyPI, on demand.

ReadTheDocs builds the documentation from ``.readthedocs.yaml``, installing the package
with the ``docs`` extra on Python 3.12.

Releasing
=========

Versions come from git tags through ``setuptools_scm``; never edit
``src/polymath/_version.py`` by hand. To release, tag the commit on ``main`` with the
version, push the tag, and create a GitHub Release from it. Publishing the release
triggers the upload to PyPI. A build from a commit that is not tagged carries a
development version derived from the most recent tag.

Contributing Changes
====================

Work on a branch named ``<initials>_<YYMMDD>_<topic>``, such as ``rf_251204_mixins``.
Commit subjects are plain capitalized imperative sentences with no type prefix and no
trailing period. Every pull request must pass the full check set, and pull requests are
squash-merged onto ``main``, which appends the pull request number to the subject.
:doc:`/contributing` covers reporting bugs, proposing enhancements, and the legal terms
of a contribution.
