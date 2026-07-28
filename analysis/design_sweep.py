#!/usr/bin/env python3
"""Does a within-task PAIRED design beat an INDEPENDENT one, at equal budget?

The Stage 0 result — that pairing bought nothing — was measured by
comparing two *estimators* on the same dataset. That is the wrong
comparison. The question a practitioner actually faces is a choice
between two *designs* given a fixed number of attempts:

    paired      : n tasks, each run R times in BOTH cells
    independent : 2n tasks, half run R times in one cell, half the other

Both spend 2*n*R attempts. Which produces a tighter interval on the
contrast RTW = 1 - T_on/T_off?

Ground truth throughout is the empirical SD of RTW over many simulated
datasets — no estimator is trusted to report on itself.

Prediction under test: pairing cancels the shared token scale but NOT
the success-count noise, because successes are independent Bernoulli
draws per cell even for the same task. So the paired advantage should
be near zero at small R and grow as R rises and the per-task success
rate is estimated more precisely.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.ratio_variance import _var, _mean

TOK_OFF, TOK_ON = 1200.0, 1800.0     # 'on' genuinely costs more
P_BASE, P_ON_MULT = 0.30, 0.75       # 'on' is genuinely less accurate


def _task_params(rng, tok_sd, p_sd):
    return (math.exp(rng.gauss(0, tok_sd)),
            min(0.95, max(0.05, rng.gauss(P_BASE, p_sd))))


def _cell(rng, mult, p, base_tok, R):
    tok = [max(1.0, rng.gauss(base_tok, 180) * mult) for _ in range(R)]
    suc = [1 if rng.random() < p else 0 for _ in range(R)]
    return tok, suc


def rtw_paired(n_tasks, R, rng, tok_sd, p_sd):
    """Same tasks in both cells."""
    n_on = d_on = n_off = d_off = 0.0
    for _ in range(n_tasks):
        mult, p = _task_params(rng, tok_sd, p_sd)
        t, s = _cell(rng, mult, p, TOK_OFF, R)
        n_off += sum(t); d_off += sum(s)
        t, s = _cell(rng, mult, p * P_ON_MULT, TOK_ON, R)
        n_on += sum(t); d_on += sum(s)
    if d_on <= 0 or d_off <= 0:
        return None
    return 1.0 - (n_on / d_on) / (n_off / d_off)


def rtw_independent(n_tasks, R, rng, tok_sd, p_sd):
    """Disjoint tasks per cell, same total attempts."""
    n_on = d_on = n_off = d_off = 0.0
    for _ in range(n_tasks):          # tasks seen only in 'off'
        mult, p = _task_params(rng, tok_sd, p_sd)
        t, s = _cell(rng, mult, p, TOK_OFF, R)
        n_off += sum(t); d_off += sum(s)
    for _ in range(n_tasks):          # different tasks, only in 'on'
        mult, p = _task_params(rng, tok_sd, p_sd)
        t, s = _cell(rng, mult, p * P_ON_MULT, TOK_ON, R)
        n_on += sum(t); d_on += sum(s)
    if d_on <= 0 or d_off <= 0:
        return None
    return 1.0 - (n_on / d_on) / (n_off / d_off)


def true_se(fn, n_sims, seed, **kw):
    rng = random.Random(seed)
    vals = [v for v in (fn(rng=rng, **kw) for _ in range(n_sims))
            if v is not None]
    return math.sqrt(_var(vals)), _mean(vals), len(vals)


def sweep(tok_sd, p_sd, label, budget=320, n_sims=3000):
    print(f"\n{label}")
    print(f"  (fixed budget: {budget} attempts; n_tasks adjusts as R grows)")
    print(f"  {'R':>3} {'tasks':>6} {'SE paired':>11} {'SE indep':>10} "
          f"{'pairing gain':>13}")
    print("  " + "-" * 48)
    rows = []
    for R in (1, 2, 4, 8, 16):
        n_tasks = budget // (2 * R)
        if n_tasks < 5:
            continue
        kw = dict(n_tasks=n_tasks, R=R, tok_sd=tok_sd, p_sd=p_sd)
        se_p, _, _ = true_se(rtw_paired, n_sims, seed=11, **kw)
        se_i, _, _ = true_se(rtw_independent, n_sims, seed=12, **kw)
        gain = se_i / se_p if se_p > 0 else float("nan")
        print(f"  {R:>3} {n_tasks:>6} {se_p:>11.4f} {se_i:>10.4f} "
              f"{gain:>12.2f}x")
        rows.append((R, n_tasks, se_p, se_i, gain))
    return rows


if __name__ == "__main__":
    print("=" * 62)
    print("PAIRED vs INDEPENDENT DESIGN, equal attempt budget")
    print("=" * 62)

    a = sweep(tok_sd=0.60, p_sd=0.02,
              label="A. task difficulty is mostly a TOKEN-SCALE effect")
    b = sweep(tok_sd=0.60, p_sd=0.12,
              label="B. task difficulty affects tokens AND success rate")
    c = sweep(tok_sd=0.05, p_sd=0.12,
              label="C. task difficulty is mostly a SUCCESS-RATE effect")

    print("\n" + "=" * 62)
    print("READING")
    print("=" * 62)
    for rows, name in ((a, "A"), (b, "B"), (c, "C")):
        if not rows:
            continue
        lo, hi = rows[0][4], rows[-1][4]
        trend = "grows with R" if hi > lo * 1.15 else \
                "flat in R" if hi > lo * 0.85 else "shrinks with R"
        print(f"  {name}: gain {lo:.2f}x at R={rows[0][0]} -> "
              f"{hi:.2f}x at R={rows[-1][0]}  ({trend})")
