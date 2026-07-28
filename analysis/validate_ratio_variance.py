"""Validate the ratio-estimator variance against simulation.

The test is not "does it run" but "does it recover a variance we
independently know". Ground truth here is the EMPIRICAL sampling
distribution of T over many simulated datasets: simulate 2000 datasets
from the same generating process, compute T for each, and take the
standard deviation. Any correct SE formula must reproduce that number.
"""
import sys
import math
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.ratio_variance import (decompose, cluster_bootstrap_se,
                                     mde_relative, _var, _mean)


def simulate(n_tasks, attempts_per_task, p_success, tok_success,
             tok_failure, task_sd, rng):
    """One dataset.

    tok_failure > tok_success reproduces the realistic case: a failed
    attempt burns its full budget (truncation, retries), so failures cost
    MORE than successes and the token/success covariance is negative.
    task_sd introduces genuine between-task heterogeneity, which is what
    makes cluster resampling necessary.
    """
    tokens, successes, task_ids = [], [], []
    for t in range(n_tasks):
        task_effect = rng.gauss(1.0, task_sd)          # multiplicative
        task_p = min(0.98, max(0.02, p_success + rng.gauss(0, 0.10)))
        for _ in range(attempts_per_task):
            s = 1 if rng.random() < task_p else 0
            base = tok_success if s else tok_failure
            tokens.append(max(1.0, rng.gauss(base, base * 0.15) * task_effect))
            successes.append(s)
            task_ids.append(t)
    return tokens, successes, task_ids


def true_se_by_simulation(n_sims, seed, **kw):
    """Empirical SD of T across independently simulated datasets."""
    rng = random.Random(seed)
    ts = []
    for _ in range(n_sims):
        tok, suc, _ = simulate(rng=rng, **kw)
        if sum(suc) > 0:
            ts.append(sum(tok) / sum(suc))
    return math.sqrt(_var(ts)), _mean(ts), len(ts)


def scenario(name, n_sims=2000, **kw):
    print(f"\n{'='*66}\n{name}\n{'='*66}")
    true_se, mean_T, kept = true_se_by_simulation(n_sims, seed=99, **kw)
    print(f"  ground truth from {kept} simulated datasets:")
    print(f"    mean T   = {mean_T:,.1f} tokens per success")
    print(f"    TRUE SE  = {true_se:,.1f}")

    # one dataset, analysed the three ways
    rng = random.Random(7)
    tok, suc, tids = simulate(rng=rng, **kw)
    d = decompose(tok, suc)
    boot = cluster_bootstrap_se(tok, suc, tids, n_resamples=3000, seed=1)

    print(f"\n  estimated from ONE dataset (n={d.n} attempts, "
          f"{len(set(tids))} tasks):")
    print(f"    naive SE      = {d.naive_se:8,.1f}   "
          f"({100*d.naive_se/true_se:5.1f}% of truth)")
    print(f"    delta SE      = {d.delta_se:8,.1f}   "
          f"({100*d.delta_se/true_se:5.1f}% of truth)")
    print(f"    bootstrap SE  = {boot:8,.1f}   "
          f"({100*boot/true_se:5.1f}% of truth)")

    print(f"\n  variance decomposition (delta method):")
    print(f"    token variance     {100*d.share_tokens:6.1f}%")
    print(f"    covariance term    {100*d.share_covariance:6.1f}%"
          f"   {'(inflates)' if d.var_covariance > 0 else '(deflates)'}")
    print(f"    success-rate term  {100*d.share_success:6.1f}%")
    print(f"    -> naive interval is {100*(1-1/d.naive_understatement):.0f}% "
          f"too narrow")
    return d, true_se, boot


if __name__ == "__main__":
    base = dict(n_tasks=40, attempts_per_task=4, task_sd=0.20)

    # 1. failures cost MORE than successes — the realistic agentic case
    d1, t1, b1 = scenario(
        "A. Failures cost more than successes (realistic)",
        p_success=0.45, tok_success=1200, tok_failure=2600, **base)

    # 2. failures cost the SAME — covariance term should vanish
    d2, t2, b2 = scenario(
        "B. Failures cost the same (covariance ~ 0)",
        p_success=0.45, tok_success=1500, tok_failure=1500, **base)

    # 3. low success rate — the denominator gets dangerous
    d3, t3, b3 = scenario(
        "C. Low success rate (denominator near zero)",
        p_success=0.15, tok_success=1200, tok_failure=2600, **base)

    print(f"\n{'='*66}\nCHECKS\n{'='*66}")
    ok = True

    # the delta method must beat the naive estimator in every scenario
    for i, (d, t) in enumerate([(d1, t1), (d2, t2), (d3, t3)], 1):
        better = abs(d.delta_se - t) < abs(d.naive_se - t)
        ok &= better
        print(f"  scenario {i}: delta closer to truth than naive: {better}")

    # naive must UNDERSTATE, never overstate
    for i, d in enumerate([d1, d2, d3], 1):
        under = d.naive_se < d.delta_se
        ok &= under
        print(f"  scenario {i}: naive understates: {under} "
              f"(ratio {d.naive_understatement:.2f}x)")

    # covariance term should be ~0 when failures cost the same
    small = abs(d2.share_covariance) < 0.10
    ok &= small
    print(f"  scenario 2: covariance share ~0 when costs equal: {small} "
          f"({100*d2.share_covariance:.1f}%)")

    # covariance term should be clearly positive when failures cost more
    pos = d1.share_covariance > 0.05
    ok &= pos
    print(f"  scenario 1: covariance inflates when failures cost more: {pos} "
          f"({100*d1.share_covariance:+.1f}%)")

    print(f"\n{'ALL CHECKS PASS' if ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if ok else 1)
