#!/usr/bin/env python3
r"""At what success rate does the naive interval stop being wrong?

Stage 0 showed a naive interval on cost-per-success achieves 39% coverage
— but that was measured at one operating point, around a 30% success
rate. The dominant variance term is

    T^2 * S2x / (n * xbar^2)      with  S2x ~ p(1-p),  T ~ 1/p

so BOTH factors shrink as p rises: the Bernoulli variance collapses
toward zero near p=1, and T itself gets smaller. The naive omission
should therefore matter less and less as evaluations get easier.

That converts a blanket warning into a threshold, which is more useful:
below some success rate you must correct, above it you need not. This
sweep locates it.

Ground truth is coverage — the fraction of nominal 95% intervals that
actually contain the known population value — never a self-reported SE.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.ratio_variance import decompose, cluster_bootstrap_se, _var, _mean

N_TASKS, PER_TASK = 40, 4
TOK_SUCCESS, TOK_FAILURE = 1200.0, 2600.0   # failures cost more, as usual


def dataset(p_base, rng):
    tok, suc, tid = [], [], []
    for t in range(N_TASKS):
        mult = math.exp(rng.gauss(0, 0.20))
        p = min(0.99, max(0.01, rng.gauss(p_base, 0.08)))
        for _ in range(PER_TASK):
            s = 1 if rng.random() < p else 0
            base = TOK_SUCCESS if s else TOK_FAILURE
            tok.append(max(1.0, rng.gauss(base, base * 0.15) * mult))
            suc.append(s)
            tid.append(t)
    return tok, suc, tid


def population_T(p_base, seed=1, reps=200):
    rng = random.Random(seed)
    n = d = 0.0
    for _ in range(reps):
        tok, suc, _ = dataset(p_base, rng)
        n += sum(tok)
        d += sum(suc)
    return n / d


def sweep(p_base, n_sims=400, seed=7):
    T_pop = population_T(p_base)
    rng = random.Random(seed)
    cov_naive = cov_delta = 0
    shares, mdes, ratios, kept = [], [], [], 0
    for i in range(n_sims):
        tok, suc, tid = dataset(p_base, rng)
        if sum(suc) < 2:
            continue
        kept += 1
        d = decompose(tok, suc)
        if d.T - 1.96 * d.naive_se <= T_pop <= d.T + 1.96 * d.naive_se:
            cov_naive += 1
        if d.T - 1.96 * d.delta_se <= T_pop <= d.T + 1.96 * d.delta_se:
            cov_delta += 1
        shares.append(d.share_success)
        ratios.append(d.delta_se / d.naive_se)
        if i < 60:      # bootstrap is the expensive one; sample it
            b = cluster_bootstrap_se(tok, suc, tid, n_resamples=400, seed=i)
            if b > 0:
                mdes.append((1.96 + 0.84) * math.sqrt(2) * b / d.T)
    return dict(p=p_base, T=T_pop, kept=kept,
                naive=cov_naive / kept, delta=cov_delta / kept,
                share_success=_mean(shares), widen=_mean(ratios),
                mde=_mean(mdes) if mdes else float("nan"))


if __name__ == "__main__":
    print("=" * 78)
    print("HOW NAIVE INTERVAL ERROR DEPENDS ON THE SUCCESS RATE")
    print(f"({N_TASKS} tasks x {PER_TASK} attempts; nominal 95% intervals)")
    print("=" * 78)
    print(f"{'success':>8} {'T':>9} {'naive':>8} {'delta':>8} "
          f"{'success-rate':>13} {'correct SE':>11} {'MDE':>8}")
    print(f"{'rate':>8} {'':>9} {'coverage':>8} {'coverage':>8} "
          f"{'share of var':>13} {'vs naive':>11} {'':>8}")
    print("-" * 78)

    rows = []
    for p in (0.10, 0.20, 0.30, 0.50, 0.70, 0.85, 0.95):
        r = sweep(p)
        rows.append(r)
        print(f"{r['p']:>7.0%} {r['T']:>9,.0f} {r['naive']:>7.0%} "
              f"{r['delta']:>7.0%} {100*r['share_success']:>12.0f}% "
              f"{r['widen']:>10.2f}x {r['mde']:>7.0%}")

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    ok = [r for r in rows if r["naive"] >= 0.90]
    if ok:
        thr = min(r["p"] for r in ok)
        print(f"  Naive intervals reach >=90% coverage only at success rates")
        print(f"  of {thr:.0%} and above. Below that they understate")
        print(f"  uncertainty materially and the correction is not optional.")
    else:
        print("  Naive intervals never reach 90% coverage anywhere in this")
        print("  range — the correction is always required.")
    lo, hi = rows[0], rows[-1]
    print(f"\n  Across the range, the success-rate term falls from "
          f"{100*lo['share_success']:.0f}% to {100*hi['share_success']:.0f}% "
          f"of variance,")
    print(f"  and the minimum detectable effect from {lo['mde']:.0%} to "
          f"{hi['mde']:.0%}.")
    print(f"\n  The practical consequence: a hard evaluation is not merely")
    print(f"  harder to pass, it is harder to MEASURE. Precision and")
    print(f"  difficulty are coupled through the same denominator.")
