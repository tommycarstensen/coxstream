# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-06-15

### Added

- Prebuilt binary wheels, published automatically via cibuildwheel, for CPython
  3.10-3.14 on Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows
  (AMD64). `pip install coxstream` no longer needs a C compiler on those
  platforms.

### Changed

- No library code changes; behaviour is identical to 0.1.0.

## [0.1.0] - 2026-06-14

First public release.

### Added

- `CoxStream`: exact Efron partial-likelihood Cox proportional hazards
  estimator. Computes the score and observed information in a single
  descending-time pass per Newton-Raphson iteration, with O(p^2) working memory
  independent of the number of observations. The streamed estimate reproduces
  the in-memory maximum-likelihood estimate, and the Efron tie correction is
  carried across chunk boundaries.
- `CoxStream.fit`: fit from in-memory NumPy arrays.
- `CoxStream.fit_parquet`: out-of-core fit from a Parquet file pre-sorted by
  descending event time, never materialising the cohort (optional `[parquet]`
  extra: pyarrow).
- `check_sorted`: footer-only dry run that validates a Parquet file is sorted by
  descending event time without fitting, so a sort mistake fails fast instead of
  yielding a silently wrong fit.
- Vendored Cython kernel (a C compiler is required to build).
- Dependency-free test suite validating exactness against a plain-NumPy Cox
  Newton-Raphson reference.

[0.1.1]: https://github.com/tommycarstensen/coxstream/releases/tag/v0.1.1
[0.1.0]: https://github.com/tommycarstensen/coxstream/releases/tag/v0.1.0
