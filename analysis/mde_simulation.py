#!/usr/bin/env python3
r"""Is the normal-approximation MDE trustworthy at low success counts?

Every minimum detectable effect in FINDINGS.md comes from the closed
form

    MDE = (z_alpha + z_beta) * sqrt(2) * SE / T

which assumes the estimator is approximately normal. For a RATIO whose
denominator is a small count of successes, that assumption is doing
real work — and the report says so in its limits rather than checking
it. This checks it.

Ground truth is the definition of power, not another formula: plant a
true effect, simulate the whole procedure many times, and measure how
often the interval actually excludes zero. Then find the effect size
where that rate hits 80% and compare it with what the closed form
predicts.

If the closed form is optimistic at low success counts, then every MDE
in the report understates how hard detection really is, and the
findings get stronger rather than weaker.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.ratio_variance import decompose, _var, _mean

TOK_OFF, TOK_FAIL = 1200.0, 2600.0
TASK_SD = 0.20


def paired_dataset(n_tasks, per_cell, p, ratio, rng):
    """Two cells over the same tasks; 'on' costs `ratio` times as much."""
    on, off = ([], []), ([], [])
    for _ in range(n_tasks):
        mult = math.exp(rng.gauss(0, TASK_SD))
        p_t = min(0.99, max(0.01, rng.gauss(p, 0.08)))
        for _ in range(per_cell):
            s = 1 if rng.random() < p_t else 0
            off[0].append(max(1., rng.gauss(TOK_OFF if s else TOK_FAIL, 200) * mult))
            off[1].append(s)
            s2 = 1 if rng.random() < p_t else 0
            on[0].append(max(1., rng.gauss((TOK_OFF if s2 else TOK_FAIL) * ratio,
                                           200) * mult))
            on[1].append(s2)
    return on, off


def rtw_and_se(on, off):
    """Contrast plus a delta-method SE for it, propagating both arms."""
    if sum(on[1]) < 2 or sum(off[1]) < 2:
        return None, None
    d_on, d_off = decompose(*on), decompose(*off)
    rtw = 1 - d_on.T / d_off.T
    rel = math.sqrt((d_on.delta_se / d_on.T) ** 2 +
                    (d_off.delta_se / d_off.T) ** 2)
    return rtw, (d_on.T / d_off.T) * rel


def empirical_power(n_tasks, per_cell, p, ratio, n_sims, seed):
    """Fraction of simulated studies whose interval excludes zero."""
    rng = random.Random(seed)
    hits = usable = 0
    for _ in range(n_sims):
        on, off = paired_dataset(n_tasks, per_cell, p, ratio, rng)
        rtw, se = rtw_and_se(on, off)
        if rtw is None or se is None or se <= 0:
            continue
        usable += 1
        if abs(rtw) - 1.96 * se > 0:
            hits += 1
    return (hits / usable if usable else 0.0), usable


def closed_form_mde(n_tasks, per_cell, p, seed, reps=60):
    """What the report's formula predicts at this operating point."""
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        on, off = paired_dataset(n_tasks, per_cell, p, 1.0, rng)
        _, se = rtw_and_se(on, off)
        if se:
            vals.append((1.96 + 0.84) * se)
    return _mean(vals) if vals else float("nan")


def simulated_mde(n_tasks, per_cell, p, seed, n_sims=400, target=0.80):
    """Smallest true effect detected at `target` power, found by search
    over the actual procedure."""
    lo, hi = 0.02, 6.0
    for _ in range(14):
        mid = (lo + hi) / 2
        pw, usable = empirical_power(n_tasks, per_cell, p, 1 + mid, n_sims, seed)
        if usable < n_sims * 0.5:      # too few estimable studies
            return None
        if pw >= target:
            hi = mid
        else:
            lo = mid
    val = (lo + hi) / 2
    # If the search converged on its own ceiling, the target was never
    # reached: report a LOWER BOUND rather than a point estimate.
    return (val, val > 5.9)


if __name__ == "__main__":
    print("=" * 74)
    print("NORMAL-APPROXIMATION MDE vs MDE MEASURED BY SIMULATION")
    print("(paired design, 2 attempts per task per cell, 80% power)")
    print("=" * 74)
    print(f"{'success':>8} {'tasks':>6} {'successes':>10} "
          f"{'closed form':>12} {'simulated':>10} {'ratio':>7}")
    print(f"{'rate':>8} {'':>6} {'per cell':>10} {'':>12} {'':>10} {'':>7}")
    print("-" * 74)

    rows = []
    for p, n_tasks in ((0.10, 40), (0.20, 40), (0.30, 40),
                       (0.50, 40), (0.30, 13), (0.15, 13)):
        cf = closed_form_mde(n_tasks, 2, p, seed=5)
        res = simulated_mde(n_tasks, 2, p, seed=9)
        sm, at_bound = (None, False) if res is None else res
        succ = n_tasks * 2 * p
        if sm is None:
            print(f"{p:>7.0%} {n_tasks:>6} {succ:>10.0f} "
                  f"{cf:>11.0%} {'not est.':>10} {'—':>7}")
            continue
        mark = ">=" if at_bound else " "
        print(f"{p:>7.0%} {n_tasks:>6} {succ:>10.0f} "
              f"{cf:>11.0%} {mark}{sm:>8.0%} {mark}{sm/cf:>5.2f}x")
        rows.append((p, n_tasks, succ, cf, sm, at_bound))

    print("\n" + "=" * 74)
    if rows:
        exact = [r for r in rows if not r[5]] or rows
        worst = max(exact, key=lambda r: r[4] / r[3])
        print(f"Largest discrepancy: at {worst[0]:.0%} success with "
              f"{worst[2]:.0f} successes per cell, the true detectable")
        print(f"effect is {worst[4]/worst[3]:.2f}x what the closed form "
              f"predicts ({worst[4]:.0%} vs {worst[3]:.0%}).")
        if any(r[5] for r in rows):
            print("\nRows marked >= hit the search ceiling: 80% power was")
            print("never reached at any effect size tested, so those are")
            print("lower bounds and the true value may be far higher.")
        low = [r for r in rows if r[2] < 10]
        if low and _mean([r[4] / r[3] for r in low]) > 1.15:
            print("\nThe closed form is OPTIMISTIC where successes are scarce.")
            print("Every MDE reported from it therefore understates how hard")
            print("detection actually is — the findings are conservative, not")
            print("inflated.")
