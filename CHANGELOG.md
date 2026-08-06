# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-05

### Added

- `CoxStream.fit` now accepts the follow-up time as a `start`/`stop` pair
  (`durations = stop - start`) as an alternative to a precomputed `durations`
  array, matching the ergonomics of survival packages that take interval
  endpoints. This is right-censored follow-up only: entry times are not modelled
  (no left-truncation / counting-process risk sets).

### Fixed

- `fit_parquet` now streams the parquet one row group at a time
  (`read_row_group`) instead of `pyarrow.iter_batches`. `iter_batches`
  read-ahead-buffers proportional to the file size, so peak RSS grew with the
  number of rows -- which defeats the whole point of out-of-core fitting (on a
  Linux VM the footprint climbed from ~0.4 GB toward the full-file size as n
  grew; macOS memory compression had hidden it). Reading row groups
  individually keeps peak at O(row_group * p), independent of the cohort size
  (verified flat to 32M rows). Peak is now set by the input's largest row
  group, so write the sorted parquet with modest row groups.

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

[0.2.0]: https://github.com/tommycarstensen/coxstream/releases/tag/v0.2.0
[0.1.1]: https://github.com/tommycarstensen/coxstream/releases/tag/v0.1.1
[0.1.0]: https://github.com/tommycarstensen/coxstream/releases/tag/v0.1.0
