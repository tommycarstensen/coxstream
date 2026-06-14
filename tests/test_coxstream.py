"""Tests for coxstream.

The core tests need only the ``test`` extra (pytest). The lifelines exactness
cross-check additionally needs the ``verify`` extra and is skipped otherwise
(``importorskip``):

    pip install -e '.[test,verify]'
    pytest
"""
import numpy as np
import pytest

from coxstream import CoxStream


def _simulate(n=20_000, p=4, seed=0):
    """Weibull proportional-hazards data with known coefficients."""
    rng = np.random.default_rng(seed)
    beta = np.array([0.5, -0.4, 0.3, -0.2])[:p]
    X = rng.standard_normal((n, p))
    eta = X @ beta
    u = rng.uniform(size=n)
    t_event = (-np.log(u) / np.exp(eta)) ** (1.0 / 1.5)
    t_cens = rng.exponential(scale=t_event.mean() * 1.5, size=n)
    t = np.minimum(t_event, t_cens)
    e = (t_event <= t_cens).astype(int)
    return t, e, X, beta


def test_recovers_known_coefficients():
    t, e, X, beta = _simulate()
    model = CoxStream().fit(t, e, X)
    assert model.coef_ is not None
    assert model.coef_.shape == (X.shape[1],)
    assert model.n_iter_ >= 1
    # Sampling error at n=20k, p=4 is well under 0.1.
    assert np.max(np.abs(model.coef_ - beta)) < 0.1


def test_batch_size_invariant():
    """The estimate must not depend on the streaming batch size."""
    t, e, X, _ = _simulate()
    a = CoxStream(batch_size=512).fit(t, e, X).coef_
    b = CoxStream(batch_size=50_000).fit(t, e, X).coef_
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_handles_ties():
    """Discretised (heavily tied) times still fit and recover coefficients."""
    t, e, X, beta = _simulate()
    t_tied = np.ceil(t * 4) / 4  # quarter-unit grid -> many ties
    model = CoxStream().fit(t_tied, e, X)
    assert np.max(np.abs(model.coef_ - beta)) < 0.15


@pytest.mark.parametrize("bad", ["1d_X", "mismatched"])
def test_input_validation(bad):
    t, e, X, _ = _simulate(n=100)
    with pytest.raises(ValueError):
        if bad == "1d_X":
            CoxStream().fit(t, e, X[:, 0])
        else:
            CoxStream().fit(t[:50], e, X)


def test_fit_parquet_matches_fit(tmp_path):
    """The out-of-core path (pre-sorted Parquet) must equal the in-memory fit."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    t, e, X, _ = _simulate(n=8_000, p=3)
    t = np.ceil(t * 4) / 4  # induce ties so the tie path runs through disk
    cols = [f"x{i}" for i in range(X.shape[1])]

    # fit_parquet requires the file pre-sorted by descending event time.
    order = np.argsort(t, kind="stable")[::-1]
    table = pa.table({
        "duration": t[order],
        "event": e[order],
        **{c: X[order, i] for i, c in enumerate(cols)},
    })
    path = tmp_path / "cohort_desc.parquet"
    pq.write_table(table, path)

    ref = CoxStream().fit(t, e, X).coef_
    model = CoxStream().fit_parquet(str(path), "duration", "event", cols)
    np.testing.assert_allclose(model.coef_, ref, atol=1e-8)
    assert model.n_obs_ == len(t)
    assert model.feature_names_ == cols


def test_fit_parquet_rejects_unsorted(tmp_path):
    """An ascending (i.e. not DESC) Parquet is rejected from footer stats alone,
    and assume_sorted=True bypasses the check."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    t, e, X, _ = _simulate(n=8_000, p=3)
    cols = [f"x{i}" for i in range(X.shape[1])]
    order = np.argsort(t, kind="stable")            # ASCENDING -> wrong order
    table = pa.table({
        "duration": t[order],
        "event": e[order],
        **{c: X[order, i] for i, c in enumerate(cols)},
    })
    path = tmp_path / "cohort_asc.parquet"
    # Multiple row groups so the cross-group footer check has something to compare.
    pq.write_table(table, path, row_group_size=2_000)

    with pytest.raises(ValueError, match="not sorted by descending"):
        CoxStream().fit_parquet(str(path), "duration", "event", cols)
    # Opt-out runs without raising (result is meaningless, but the guard is off).
    CoxStream().fit_parquet(str(path), "duration", "event", cols,
                            assume_sorted=True)


def test_check_sorted_dry_run(tmp_path):
    """The public dry-run validator agrees with fit_parquet's built-in guard."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq
    from coxstream import check_sorted

    t, e, X, _ = _simulate(n=8_000, p=2)
    order = np.argsort(t, kind="stable")

    asc = tmp_path / "asc.parquet"
    pq.write_table(pa.table({"duration": t[order], "event": e[order]}),
                   asc, row_group_size=2_000)
    with pytest.raises(ValueError, match="not sorted by descending"):
        check_sorted(str(asc), "duration")

    desc = tmp_path / "desc.parquet"
    pq.write_table(pa.table({"duration": t[order][::-1], "event": e[order][::-1]}),
                   desc, row_group_size=2_000)
    assert check_sorted(str(desc), "duration") is None  # passes, returns None


def _cox_nr_reference(t, e, X, max_iter=50, tol=1e-10):
    """Independent plain-numpy exact Cox partial-likelihood Newton-Raphson.

    Risk-set moments via reverse cumulative sums over descending-time order.
    Assumes distinct event times (Efron then equals the exact partial
    likelihood), so it is a clean oracle for continuous simulated data without
    any third-party dependency.
    """
    order = np.argsort(t)[::-1]              # descending time
    e = e[order].astype(bool)
    X = X[order]
    _, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        w = np.exp(X @ beta)
        S0 = np.cumsum(w)                                    # risk-set suffix sum
        S1 = np.cumsum(w[:, None] * X, axis=0)
        outer = X[:, :, None] * X[:, None, :]
        S2 = np.cumsum(w[:, None, None] * outer, axis=0)
        m = S1[e] / S0[e, None]
        score = (X[e] - m).sum(0)
        hess = (S2[e] / S0[e, None, None]
                - m[:, :, None] * m[:, None, :]).sum(0)
        step = np.linalg.solve(hess, score)
        beta = beta + step
        if np.linalg.norm(step) < tol:
            break
    return beta


def test_matches_numpy_reference():
    """Exactness: CoxStream reproduces an independent numpy MLE (distinct times)."""
    t, e, X, _ = _simulate(n=5_000, p=3)   # continuous Weibull -> distinct times
    ref = _cox_nr_reference(t, e, X)
    coef = CoxStream().fit(t, e, X).coef_
    np.testing.assert_allclose(coef, ref, atol=1e-6)
