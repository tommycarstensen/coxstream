# Contributing to coxstream

Thanks for your interest in improving `coxstream`. Bug reports, fixes, and
focused features are welcome.

## Reporting issues

Open an issue with: what you ran, what you expected, what happened, and a
minimal reproducer (ideally a small array example). Include your OS, Python
version, and `coxstream`/`numpy` versions.

## Development setup

`coxstream` builds a small Cython kernel, so you need a C compiler.

```bash
git clone <repo-url>
cd coxstream
pip install -e '.[test,parquet]'   # build + tests + out-of-core (pyarrow) deps
```

Re-run `pip install -e .` after editing `src/coxstream/_kernel.pyx` so the
kernel is recompiled.

## Tests and linting

```bash
pytest                 # full suite (recovery, batch-invariance, ties, out-of-core)
ruff check .           # style / lint
```

All tests must pass and `ruff check` must be clean before a pull request. New
behaviour needs a test; numerical changes should keep the exactness check
(`test_matches_numpy_reference`, an independent plain-numpy Cox NR) green to
~1e-6.

## Scope

`coxstream` is deliberately a focused estimator: exact Efron Cox regression,
out of core. Proposals that broaden it into a general survival suite (baseline
hazard, time-varying covariates, PH diagnostics) are likely out of scope --
please open an issue to discuss before implementing.

## Pull requests

- One logical change per PR; keep the diff focused.
- Match the existing style (PEP 8, ruff-clean, numpy-only runtime).
- Update the README / docstrings if you change the public API.
- By contributing you agree your work is licensed under the project's MIT
  license.
