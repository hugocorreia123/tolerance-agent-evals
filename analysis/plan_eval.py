#!/usr/bin/env python3
r"""How big does an evaluation need to be?

Everything else in this repository is diagnostic: it takes runs you have
already paid for and tells you what they could resolve. This inverts the
question into the one worth asking BEFORE spending the compute —

    "I want to detect a 20% change in cost per success.
     How many tasks do I need?"

Method: no closed form and no extrapolated scaling law. For each
candidate suite size, simulate datasets from the stated operating point,
compute the delta-method standard error (validated in
validate_ratio_variance.py at ~95% of truth), convert to a minimum
detectable effect, and bisect on the smallest size that meets the
target. The scaling is measured, not assumed.

    python3 analysis/plan_eval.py --target 0.20 --success-rate 0.30
    python3 analysis/plan_eval.py --table
"""
import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.ratio_variance import decompose, mde_relative, _mean

TOK_SUCCESS, TOK_FAILURE, TASK_SD = 1200.0, 2600.0, 0.20


def simulate_mde(n_tasks, per_task, p, rng, reps=25, power=0.80):
    """Average relative MDE over `reps` simulated datasets at this size."""
    out = []
    for _ in range(reps):
        tok, suc = [], []
        for _ in range(n_tasks):
            mult = math.exp(rng.gauss(0, TASK_SD))
            p_task = min(0.99, max(0.01, rng.gauss(p, 0.08)))
            for _ in range(per_task):
                s = 1 if rng.random() < p_task else 0
                base = TOK_SUCCESS if s else TOK_FAILURE
                tok.append(max(1.0, rng.gauss(base, base * 0.15) * mult))
                suc.append(s)
        if sum(suc) < 2:
            continue
        d = decompose(tok, suc)
        # delta SE understates slightly by ignoring clustering; inflate by
        # the factor measured in validation so the answer is not optimistic
        out.append(mde_relative(d.T, d.delta_se * 1.15, power=power))
    return _mean(out) if out else float("inf")


def required_tasks(target, p, per_task=2, power=0.80, cap=4000, seed=0):
    """Smallest task count meeting the target MDE. Returns None if the
    target is unreachable within `cap` tasks — which is itself an
    answer, and often the right one."""
    rng = random.Random(seed)
    lo, hi = 5, 40
    while hi <= cap and simulate_mde(hi, per_task, p, rng, power=power) > target:
        lo, hi = hi, hi * 2
    if hi > cap:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if simulate_mde(mid, per_task, p, rng, power=power) <= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.20,
                    help="effect you want to detect, as a fraction of "
                         "current cost per success (0.20 = 20%%)")
    ap.add_argument("--success-rate", type=float, default=0.30)
    ap.add_argument("--replicates", type=int, default=2,
                    help="attempts per task per condition")
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--table", action="store_true",
                    help="print the full lookup instead of one query")
    args = ap.parse_args()

    if args.table:
        targets = (0.10, 0.20, 0.30, 0.50, 1.00)
        rates = (0.10, 0.30, 0.50, 0.70, 0.95)
        print("TASKS REQUIRED to detect a given change in cost per success")
        print(f"({args.replicates} attempts per task per condition, "
              f"{100*args.power:.0f}% power)")
        print()
        print(f"{'success':>8} " + "".join(f"{100*t:>9.0f}%" for t in targets))
        print(f"{'rate':>8} " + "".join(f"{'':>10}" for _ in targets))
        print("-" * (9 + 10 * len(targets)))
        for p in rates:
            row = f"{p:>7.0%} "
            for t in targets:
                n = required_tasks(t, p, args.replicates, args.power)
                row += f"{'>4000' if n is None else n:>10}"
            print(row)
        print()
        print("Read a column downward: the same target gets dramatically")
        print("cheaper as the success rate rises. Read a row rightward: a")
        print("coarser claim is dramatically cheaper to support.")
        print()
        print("'>4000' means the target is out of reach at any suite size")
        print("a solo researcher would build. That is a real answer — it")
        print("says the claim cannot be made, not that you need more data.")
        return

    n = required_tasks(args.target, args.success_rate, args.replicates,
                       args.power)
    print(f"target                : detect a {args.target:.0%} change in "
          f"cost per success")
    print(f"assumed success rate  : {args.success_rate:.0%}")
    print(f"attempts per task/cond: {args.replicates}")
    print(f"power                 : {args.power:.0%}")
    print()
    if n is None:
        print("REQUIRED TASKS        : more than 4,000")
        print()
        print("At this success rate the target is not reachable at any")
        print("realistic suite size. Options, in order of cost: raise the")
        print("success rate (easier tasks or a stronger model), accept a")
        print("coarser claim, or do not make the claim.")
    else:
        total = n * args.replicates * 2
        print(f"REQUIRED TASKS        : {n:,}")
        print(f"  -> {total:,} attempts across both conditions")
        print()
        print("Raising the success rate is usually cheaper than raising the")
        print("task count: the denominator drives the variance (see §4.6).")


if __name__ == "__main__":
    main()
