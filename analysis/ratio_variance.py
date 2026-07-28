r"""Ratio-estimator variance for agentic cost-efficiency metrics.

The headline metric of an agentic eval is usually

    T  =  (total tokens spent, including on failures)  /  (successes)

This is a RATIO ESTIMATOR: both numerator and denominator are random.
Its variance is therefore NOT the variance of the numerator scaled by a
constant, which is what you get if you treat the success count as fixed.

By the delta method, for T = ybar / xbar with y = tokens per attempt and
x = success indicator (0/1):

    Var(T) ~=  1/(n * xbar^2) * [ S2y  -  2*T*Sxy  +  T^2 * S2x ]
                                  \___/    \_____/    \________/
                                 tokens   covariance  success-rate

Three consequences, each measurable:

1. The success-rate term is strictly positive and is omitted entirely by
   the naive treatment.
2. The covariance term is signed. When failures cost MORE than successes
   — the usual case, because failures burn a full budget and truncate —
   Sxy is negative, so -2*T*Sxy is POSITIVE and inflates variance further.
3. Both omissions push the same way, so a naive interval is too narrow.

This module quantifies all three, and compares three standard errors:

    naive     ratio-blind, cluster-blind  (what most reports use)
    delta     ratio-aware, cluster-blind
    bootstrap ratio-aware, cluster-aware  (the reference)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from statistics import NormalDist


@dataclass
class VarianceDecomposition:
    """All quantities are on the variance scale unless named *_se."""
    T: float                 # tokens per success
    n: int                   # attempts
    success_rate: float

    var_tokens: float        # S2y  / (n xbar^2)
    var_covariance: float    # -2 T Sxy / (n xbar^2)   (signed)
    var_success: float       # T^2 S2x / (n xbar^2)
    var_total_delta: float   # sum of the three

    naive_se: float
    delta_se: float

    @property
    def share_tokens(self) -> float:
        return self.var_tokens / self.var_total_delta

    @property
    def share_covariance(self) -> float:
        return self.var_covariance / self.var_total_delta

    @property
    def share_success(self) -> float:
        return self.var_success / self.var_total_delta

    @property
    def naive_understatement(self) -> float:
        """How much wider the correct SE is than the naive one.
        1.40 means the naive interval is ~29% too narrow."""
        return self.delta_se / self.naive_se if self.naive_se > 0 else float("nan")

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(share_tokens=self.share_tokens,
                 share_covariance=self.share_covariance,
                 share_success=self.share_success,
                 naive_understatement=self.naive_understatement)
        return d


def _mean(v):
    return sum(v) / len(v)


def _var(v, mean=None):
    """Sample variance, n-1 denominator."""
    m = _mean(v) if mean is None else mean
    n = len(v)
    if n < 2:
        return 0.0
    return sum((x - m) ** 2 for x in v) / (n - 1)


def _cov(a, b):
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = _mean(a), _mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (n - 1)


def decompose(tokens: list, successes: list) -> VarianceDecomposition:
    """Delta-method variance decomposition for T = sum(tokens)/sum(successes).

    tokens    : per-attempt total tokens (in + out), failures included
    successes : per-attempt 0/1 outcome
    """
    if len(tokens) != len(successes):
        raise ValueError("tokens and successes must be the same length")
    n = len(tokens)
    if n < 2:
        raise ValueError("need at least 2 attempts")

    xbar = _mean(successes)
    if xbar <= 0:
        raise ValueError("no successes — T is undefined")
    ybar = _mean(tokens)
    T = ybar / xbar

    s2y = _var(tokens, ybar)
    s2x = _var(successes, xbar)
    sxy = _cov(tokens, successes)

    scale = 1.0 / (n * xbar ** 2)
    var_tokens = scale * s2y
    var_cov = scale * (-2.0 * T * sxy)
    var_success = scale * (T ** 2) * s2x
    var_total = var_tokens + var_cov + var_success

    # The naive treatment: pretend the denominator is fixed, so only the
    # numerator's variance propagates.
    naive_se = math.sqrt(max(var_tokens, 0.0))
    delta_se = math.sqrt(max(var_total, 0.0))

    return VarianceDecomposition(
        T=T, n=n, success_rate=xbar,
        var_tokens=var_tokens, var_covariance=var_cov,
        var_success=var_success, var_total_delta=var_total,
        naive_se=naive_se, delta_se=delta_se)


def cluster_bootstrap_se(tokens: list, successes: list, task_ids: list,
                         n_resamples: int = 4000, seed: int = 0) -> float:
    """Cluster bootstrap over task ids — the reference SE.

    Resamples TASKS with replacement, never individual attempts: attempts
    within a task are correlated, and resampling rows would understate
    variance for the same structural reason the naive SE does.
    """
    by_task = {}
    for tok, suc, tid in zip(tokens, successes, task_ids):
        by_task.setdefault(tid, []).append((tok, suc))
    ids = list(by_task)
    rng = random.Random(seed)
    out = []
    for _ in range(n_resamples):
        num = den = 0.0
        for tid in rng.choices(ids, k=len(ids)):
            for tok, suc in by_task[tid]:
                num += tok
                den += suc
        if den > 0:
            out.append(num / den)
    if len(out) < 2:
        return float("nan")
    return math.sqrt(_var(out))


def mde(se: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable difference between two independent arms, on the
    scale of T. Uses the normal approximation; the SE of a difference of
    two independent arms is sqrt(2)*se when both have the same SE."""
    nd = NormalDist()          # stdlib — no scipy, no numpy, no install
    z_a = nd.inv_cdf(1 - alpha / 2)
    z_b = nd.inv_cdf(power)
    return (z_a + z_b) * math.sqrt(2.0) * se


def mde_relative(T: float, se: float, **kw) -> float:
    """MDE expressed as a fraction of T — the practitioner-facing number:
    'you cannot detect a cost change smaller than this percentage'."""
    return mde(se, **kw) / T


def paired_contrast_bootstrap(on_tokens, on_succ, on_tasks,
                              off_tokens, off_succ, off_tasks,
                              n_resamples: int = 4000, seed: int = 0):
    """SE of RTW = 1 - T_on/T_off for a WITHIN-TASK paired design.

    Why this is not sqrt(2) * SE(cell): in a within-task factorial the
    same tasks appear in both cells, so the (large) between-task
    component is common to numerator and denominator and cancels out of
    the contrast. Treating the arms as independent therefore overstates
    the SE of the difference — badly, when between-task variance
    dominates, which the decomposition shows it does.

    Resamples TASKS once and applies the same draw to both cells, which
    is what preserves the pairing.
    """
    on_by, off_by = {}, {}
    for tok, s, t in zip(on_tokens, on_succ, on_tasks):
        on_by.setdefault(t, []).append((tok, s))
    for tok, s, t in zip(off_tokens, off_succ, off_tasks):
        off_by.setdefault(t, []).append((tok, s))

    shared = sorted(set(on_by) & set(off_by))
    if len(shared) < 3:
        return float("nan"), float("nan"), 0

    def rtw(task_draw):
        n_on = d_on = n_off = d_off = 0.0
        for t in task_draw:
            for tok, s in on_by[t]:
                n_on += tok; d_on += s
            for tok, s in off_by[t]:
                n_off += tok; d_off += s
        if d_on <= 0 or d_off <= 0 or n_off <= 0:
            return None
        return 1.0 - (n_on / d_on) / (n_off / d_off)

    point = rtw(shared)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_resamples):
        v = rtw(rng.choices(shared, k=len(shared)))
        if v is not None:
            draws.append(v)
    if len(draws) < 2:
        return point, float("nan"), len(shared)
    return point, math.sqrt(_var(draws)), len(shared)


def unpaired_contrast_se(se_on: float, se_off: float, T_on: float, T_off: float):
    """What you'd get treating the arms as independent, on the RTW scale.
    Kept only to quantify what the pairing buys."""
    if T_off <= 0:
        return float("nan")
    # RTW = 1 - T_on/T_off ; delta method on the quotient
    rel = math.sqrt((se_on / T_on) ** 2 + (se_off / T_off) ** 2)
    return (T_on / T_off) * rel
