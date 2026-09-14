"""Statistics for seed-level comparisons.

The unit of analysis is the seed (one training run, evaluated on fixed seeds).
Point estimates use the interquartile mean, intervals use a stratified
percentile bootstrap (Agarwal et al. 2021), and comparisons use the
probability of improvement, paired Wilcoxon and permutation tests, with Holm
correction within each family.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from hamlet.metrics.common import DEFAULTS, default_rng


def iqm(x: np.ndarray) -> float:
    """Interquartile mean: the mean of the middle 50% (25% trimmed each side)."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        return float("nan")
    return float(sps.trim_mean(arr, 0.25))


def _strata(x: np.ndarray | Sequence[np.ndarray]) -> list[np.ndarray]:
    """Split the input into strata: columns of a 2-D array, elements of a list, or one stratum."""
    if isinstance(x, (list, tuple)):
        return [np.asarray(s, dtype=np.float64).ravel() for s in x]
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 2:
        return [arr[:, j] for j in range(arr.shape[1])]
    return [arr.ravel()]


def bootstrap_ci(
    x: np.ndarray | Sequence[np.ndarray],
    stat: Callable[[np.ndarray], float] = iqm,
    n: int | None = None,
    rng: np.random.Generator | None = None,
    alpha: float | None = None,
) -> tuple[float, float, float]:
    """Stratified percentile bootstrap interval ``(low, high, point)`` for ``stat``.

    ``x`` is a 1-D array of per-run values (one stratum), a 2-D array with
    runs in rows and strata (tasks or conditions) in columns, or a list of
    per-stratum arrays. Runs are resampled with replacement inside each
    stratum and ``stat`` is applied to the pooled sample.
    """
    rng = default_rng(rng)
    n = DEFAULTS.n_boot if n is None else int(n)
    alpha = DEFAULTS.alpha if alpha is None else float(alpha)
    strata = _strata(x)
    pooled = np.concatenate(strata)
    point = float(stat(pooled))
    draws = np.empty(n, dtype=np.float64)
    for b in range(n):
        sample = np.concatenate([s[rng.integers(0, s.size, size=s.size)] for s in strata if s.size])
        draws[b] = stat(sample)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high), point


def prob_improvement_value(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) with ties counted as one half (the Mann-Whitney U over n*m)."""
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    diff = xa[:, None] - ya[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def prob_improvement(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int | None = None,
    rng: np.random.Generator | None = None,
    alpha: float | None = None,
) -> dict[str, float]:
    """Probability of improvement P(X > Y) with a percentile bootstrap interval.

    Equal to the Vargha-Delaney A12 statistic. ``x`` and ``y`` are resampled
    independently. Returns ``{"value", "ci_low", "ci_high"}``.
    """
    rng = default_rng(rng)
    n_boot = DEFAULTS.n_boot if n_boot is None else int(n_boot)
    alpha = DEFAULTS.alpha if alpha is None else float(alpha)
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    value = prob_improvement_value(xa, ya)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        xs = xa[rng.integers(0, xa.size, size=xa.size)]
        ys = ya[rng.integers(0, ya.size, size=ya.size)]
        draws[b] = prob_improvement_value(xs, ys)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"value": value, "ci_low": float(low), "ci_high": float(high)}


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta = P(X > Y) - P(X < Y) in [-1, 1]."""
    return 2.0 * prob_improvement_value(x, y) - 1.0


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, monotone and capped at 1."""
    p = np.asarray(pvals, dtype=np.float64).ravel()
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    factors = m - np.arange(m)
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(factors * p[order]))
    out = np.empty(m, dtype=np.float64)
    out[order] = adjusted_sorted
    return out


ALTERNATIVES = ("two-sided", "greater", "less")


def paired_wilcoxon(
    x: np.ndarray,
    y: np.ndarray | float = 0.0,
    shift: float = 0.0,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Wilcoxon signed-rank test on the paired differences ``x - y - shift``.

    ``y`` may be an array paired with ``x`` or a constant (``0.0`` by default,
    which makes this a one-sample test of ``x`` against ``shift``). ``shift``
    is the value under the null hypothesis: with ``alternative="greater"``
    the test asks whether the differences exceed ``shift``, with ``"less"``
    whether they fall short of it. This is how a non-zero threshold enters a
    p-value (a "no effect" threshold of 0.02 bits per tick is tested as
    ``shift=0.02, alternative="greater"``; a non-inferiority bound of 0.05 as
    ``shift=0.05, alternative="less"``). The exact distribution is used when
    there are no ties and at most 50 non-zero differences, which is every
    case in this study; zero differences are dropped (the ``wilcox`` rule).

    Returns ``{"statistic", "pvalue", "n", "n_nonzero", "shift"}``; p = 1
    when every shifted difference is zero.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"unknown alternative {alternative!r}; choose from {ALTERNATIVES}")
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    if ya.size == 1 and xa.size != 1:
        ya = np.full(xa.size, float(ya[0]))
    if ya.size != xa.size:
        raise ValueError(f"x and y must pair up: {xa.size} against {ya.size}")
    d = xa - ya - float(shift)
    nonzero = int(np.count_nonzero(d))
    if nonzero == 0:
        return {"statistic": 0.0, "pvalue": 1.0, "n": float(d.size), "n_nonzero": 0.0, "shift": float(shift)}
    res = sps.wilcoxon(d, alternative=alternative, zero_method="wilcox")
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue), "n": float(d.size),
            "n_nonzero": float(nonzero), "shift": float(shift)}


def sign_test(x: np.ndarray, threshold: float = 0.0, alternative: str = "greater") -> dict[str, float]:
    """Exact binomial sign test of the values of ``x`` against ``threshold``.

    Counts the values strictly above ``threshold`` among those not equal to
    it and tests that count against Binomial(n, 1/2). ``alternative``
    ``"greater"`` asks whether values tend to lie above the threshold (the
    "positive in at least k of n seeds" reading), ``"less"`` below, and
    ``"two-sided"`` either way. Values equal to the threshold are dropped.

    Returns ``{"n_above", "n", "pvalue", "threshold"}``; p = 1 when no value
    differs from the threshold.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"unknown alternative {alternative!r}; choose from {ALTERNATIVES}")
    xa = np.asarray(x, dtype=np.float64).ravel()
    d = xa - float(threshold)
    n = int(np.count_nonzero(d))
    above = int(np.sum(d > 0))
    if n == 0:
        return {"n_above": 0.0, "n": 0.0, "pvalue": 1.0, "threshold": float(threshold)}
    p = float(sps.binomtest(above, n, 0.5, alternative=alternative).pvalue)
    return {"n_above": float(above), "n": float(n), "pvalue": p, "threshold": float(threshold)}


def permutation_test_diff(
    x: np.ndarray,
    y: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_perm: int | None = None,
    rng: np.random.Generator | None = None,
    paired: bool = True,
) -> dict[str, float]:
    """Permutation test for ``stat(x) - stat(y)``.

    Paired (default): random sign flips of the differences, the statistic
    being ``stat(x - y)``. Unpaired: labels are shuffled between the pooled
    samples. The two-sided p-value counts permuted statistics at least as
    extreme as the observed one, with the observed draw included.
    """
    rng = default_rng(rng)
    n_perm = DEFAULTS.n_perm if n_perm is None else int(n_perm)
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    if paired:
        d = xa - ya
        value = float(stat(d))
        signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
        null = np.array([stat(d * s) for s in signs])
    else:
        pooled = np.concatenate([xa, ya])
        value = float(stat(xa) - stat(ya))
        null = np.empty(n_perm)
        for b in range(n_perm):
            perm = rng.permutation(pooled)
            null[b] = stat(perm[: xa.size]) - stat(perm[xa.size:])
    p = (np.sum(np.abs(null) >= abs(value)) + 1) / (n_perm + 1)
    return {"value": value, "pvalue": float(p), "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1))}


def tost(x: np.ndarray, y: np.ndarray, bound: float, alpha: float | None = None) -> dict[str, float]:
    """Two one-sided t-tests for equivalence of paired differences within ``(-bound, bound)``.

    Tests H0: mean(x - y) <= -bound and H0: mean(x - y) >= bound with paired
    one-sample t-tests; the TOST p-value is the larger of the two. ``reject``
    (1.0 or 0.0) means equivalence is concluded at level ``alpha``. The
    ``(1 - 2 alpha)`` confidence interval of the mean difference is returned.
    """
    alpha = DEFAULTS.alpha if alpha is None else float(alpha)
    d = np.asarray(x, dtype=np.float64).ravel() - np.asarray(y, dtype=np.float64).ravel()
    n = d.size
    mean = float(d.mean())
    if n < 2 or np.allclose(d, d[0]):
        inside = abs(mean) < bound
        return {"pvalue": 0.0 if inside else 1.0, "p_lower": 0.0 if inside else 1.0, "p_upper": 0.0 if inside else 1.0,
                "mean_diff": mean, "ci_low": mean, "ci_high": mean, "reject": float(inside)}
    p_lower = float(sps.ttest_1samp(d, -bound, alternative="greater").pvalue)
    p_upper = float(sps.ttest_1samp(d, bound, alternative="less").pvalue)
    p = max(p_lower, p_upper)
    half = sps.t.ppf(1 - alpha, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return {"pvalue": p, "p_lower": p_lower, "p_upper": p_upper, "mean_diff": mean,
            "ci_low": mean - float(half), "ci_high": mean + float(half), "reject": float(p < alpha)}


def per_seed(
    values: pd.DataFrame | Mapping[int, Sequence[float]],
    by: str = "seed",
    value: str = "value",
    agg: str = "median",
) -> pd.Series:
    """Collapse per-episode (or per-agent) values to one number per seed.

    ``values`` is a DataFrame with a ``by`` column and a ``value`` column, or a
    mapping from seed to an array of values. Returns a Series indexed by seed.
    """
    if isinstance(values, pd.DataFrame):
        return values.groupby(by)[value].agg(agg).sort_index()
    return pd.Series({k: pd.Series(np.asarray(v, dtype=np.float64)).agg(agg) for k, v in values.items()}).sort_index()
