"""Unit tests for the seed-level statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hamlet.metrics import stats


def test_iqm_of_known_vector():
    x = np.arange(1, 11, dtype=float)
    assert stats.iqm(x) == pytest.approx(5.5)
    assert stats.iqm(x[::-1]) == pytest.approx(5.5)
    assert np.isnan(stats.iqm([]))


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 0.5, size=16)
    low, high, point = stats.bootstrap_ci(x, n=500, rng=rng)
    assert low <= point <= high
    assert point == pytest.approx(stats.iqm(x))
    runs_by_task = rng.normal(1.0, 0.5, size=(8, 3))
    low2, high2, point2 = stats.bootstrap_ci(runs_by_task, n=500, rng=rng)
    assert low2 <= point2 <= high2
    assert point2 == pytest.approx(stats.iqm(runs_by_task.ravel()))
    low3, high3, _ = stats.bootstrap_ci([x, x + 1], stat=np.mean, n=500, rng=rng)
    assert low3 < high3


def test_prob_improvement_fully_separated_and_tied():
    rng = np.random.default_rng(1)
    x = np.arange(10, 18, dtype=float)
    y = np.arange(0, 8, dtype=float)
    res = stats.prob_improvement(x, y, n_boot=200, rng=rng)
    assert res["value"] == 1.0
    assert res["ci_low"] == 1.0 and res["ci_high"] == 1.0
    assert stats.prob_improvement_value(y, x) == 0.0
    assert stats.prob_improvement_value(x, x) == 0.5
    assert stats.cliffs_delta(x, y) == pytest.approx(1.0)
    assert stats.cliffs_delta(x, x) == pytest.approx(0.0)


def test_holm_is_monotone_and_matches_a_worked_example():
    adj = stats.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    rng = np.random.default_rng(2)
    p = rng.random(20)
    adj = stats.holm(p)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= 0)
    assert np.all(adj >= p) and np.all(adj <= 1.0)


def test_paired_wilcoxon():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, size=8)
    y = x + 2.0
    res = stats.paired_wilcoxon(y, x)
    assert res["pvalue"] < 0.05
    assert res["n"] == 8
    assert stats.paired_wilcoxon(x, x)["pvalue"] == 1.0
    # eight positive differences: exact two-sided p = 2 / 2^8, one-sided 1 / 2^8
    assert res["pvalue"] == pytest.approx(2 / 256)
    assert stats.paired_wilcoxon(y, x, alternative="greater")["pvalue"] == pytest.approx(1 / 256)
    assert stats.paired_wilcoxon(y, x, alternative="less")["pvalue"] == pytest.approx(1.0)


def test_paired_wilcoxon_shift_is_the_null_value():
    # per-seed gains of 0.05 .. 0.12 against a 0.02 threshold: every shifted difference positive
    gains = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12])
    ref = np.zeros(8)
    above = stats.paired_wilcoxon(gains, ref, shift=0.02, alternative="greater")
    assert above["pvalue"] == pytest.approx(1 / 256)
    assert above["shift"] == 0.02 and above["n_nonzero"] == 8
    # the same against a constant, and against a threshold the gains never reach
    assert stats.paired_wilcoxon(gains, 0.02, alternative="greater")["pvalue"] == pytest.approx(1 / 256)
    assert stats.paired_wilcoxon(gains, ref, shift=0.20, alternative="greater")["pvalue"] == pytest.approx(1.0)
    assert stats.paired_wilcoxon(gains, ref, shift=0.20, alternative="less")["pvalue"] == pytest.approx(1 / 256)
    # a shift that zeroes every difference gives p = 1
    same = stats.paired_wilcoxon(np.full(8, 0.02), ref, shift=0.02, alternative="greater")
    assert same["pvalue"] == 1.0 and same["n_nonzero"] == 0
    # hand-computed: differences 1, 2, 3, 4, -0.5 give W+ = 14 of 15; P(W+ >= 14) = 2 / 32 one-sided
    d = np.array([1.0, 2.0, 3.0, 4.0, -0.5])
    one = stats.paired_wilcoxon(d, np.zeros(5), alternative="greater")
    assert one["statistic"] == pytest.approx(14.0)
    assert one["pvalue"] == pytest.approx(2 / 32)
    assert stats.paired_wilcoxon(d, np.zeros(5))["pvalue"] == pytest.approx(4 / 32)
    # non-inferiority form: gaps of 0.01 .. 0.04 against a 0.05 bound, alternative "less"
    gaps = np.array([0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04])
    assert stats.paired_wilcoxon(gaps, 0.0, shift=0.05, alternative="less")["pvalue"] == pytest.approx(1 / 256)
    with pytest.raises(ValueError):
        stats.paired_wilcoxon(gaps, np.zeros(3))
    with pytest.raises(ValueError):
        stats.paired_wilcoxon(gaps, 0.0, alternative="up")


def test_sign_test_binomial_hand_cases():
    # seven of eight above zero: P(X >= 7 | n = 8, 1/2) = 9 / 256
    rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, -0.1])
    res = stats.sign_test(rho, 0.0, alternative="greater")
    assert res["n_above"] == 7 and res["n"] == 8
    assert res["pvalue"] == pytest.approx(9 / 256)
    assert stats.sign_test(np.abs(rho), 0.0)["pvalue"] == pytest.approx(1 / 256)
    # values equal to the threshold are dropped; two of two above gives 1 / 4
    assert stats.sign_test(np.array([0.0, 1.0, 2.0]), 0.0)["pvalue"] == pytest.approx(1 / 4)
    assert stats.sign_test(np.zeros(4), 0.0)["pvalue"] == 1.0
    # against a non-zero threshold and in the other direction
    z = np.array([3.0, 2.5, 2.2, 2.0, 1.0, 2.8, 2.1, 2.4])
    assert stats.sign_test(z, 1.96)["n_above"] == 7
    assert stats.sign_test(z, 1.96, alternative="less")["pvalue"] == pytest.approx(1 - 1 / 256)


def test_permutation_test_diff_paired_and_unpaired():
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, size=10)
    y = x + 3.0
    assert stats.permutation_test_diff(y, x, n_perm=500, rng=rng)["pvalue"] < 0.01
    assert stats.permutation_test_diff(y, x, n_perm=500, rng=rng, paired=False)["pvalue"] < 0.01
    same = stats.permutation_test_diff(x, x + rng.normal(0, 1e-3, size=10), n_perm=500, rng=rng)
    assert same["pvalue"] > 0.05


def test_tost_rejects_for_tiny_differences_and_not_for_large():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 1.0, size=10)
    tiny = stats.tost(x, x + rng.normal(0.0, 0.01, size=10), bound=0.5)
    assert tiny["reject"] == 1.0
    assert tiny["pvalue"] < 0.05
    assert -0.5 < tiny["ci_low"] and tiny["ci_high"] < 0.5
    large = stats.tost(x, x + 2.0 + rng.normal(0.0, 0.01, size=10), bound=0.5)
    assert large["reject"] == 0.0
    assert large["pvalue"] > 0.5


def test_per_seed_from_frame_and_mapping():
    df = pd.DataFrame({"seed": [0, 0, 1, 1, 1], "value": [1.0, 3.0, 2.0, 4.0, 9.0]})
    out = stats.per_seed(df)
    assert list(out.index) == [0, 1]
    assert out.to_numpy().tolist() == [2.0, 4.0]
    out2 = stats.per_seed({1: [2.0, 4.0, 9.0], 0: [1.0, 3.0]}, agg="mean")
    assert out2.to_numpy().tolist() == [2.0, 5.0]
