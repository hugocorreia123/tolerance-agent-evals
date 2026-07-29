#!/usr/bin/env python3
r"""Analyse judge verdicts against the objective ground truth.

Three questions, in increasing order of consequence:

  1. How often does the judge disagree with ground truth?         (error)
  2. How often does it disagree with ITSELF on identical input? (variance)
  3. Does its error rate DIFFER BY EXPERIMENTAL CONDITION?      (confound)

Only the third can invent or destroy a finding. A judge that is
uniformly wrong shifts every cell together and largely cancels in a
contrast; a judge whose error tracks the condition does not.

The report ends by recomputing the study's headline contrast under
judged outcomes, which answers the question a reader actually has:
would using an LLM judge have changed the conclusion?
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.ratio_variance import decompose, cluster_bootstrap_se, mde_relative


def load(path: Path):
    rows, manifest = [], None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("type") == "judge_manifest":
            manifest = r
        elif r.get("type") == "judge_verdict":
            rows.append(r)
    return rows, manifest


def majority(verdicts):
    """Majority verdict, ignoring unparseable replies."""
    v = [x for x in verdicts if x is not None]
    if not v:
        return None
    return sum(v) > len(v) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judge_file")
    ap.add_argument("--n-boot", type=int, default=3000)
    args = ap.parse_args()

    rows, manifest = load(Path(args.judge_file))
    if not rows:
        print("no judge verdicts found")
        sys.exit(1)

    if manifest:
        print(f"judge       : {manifest.get('judge')} "
              f"@ T={manifest.get('temperature')}, k={manifest.get('k')}")
    print(f"verdicts    : {len(rows)} attempts\n")

    # ---------- 1 & 2: error and variance, overall -------------------
    unparseable = sum(1 for r in rows if majority(r["judge_verdicts"]) is None)
    usable = [r for r in rows if majority(r["judge_verdicts"]) is not None]
    disagree = sum(1 for r in usable
                   if majority(r["judge_verdicts"]) != r["objective_success"])
    flips = sum(1 for r in usable
                if len(set(x for x in r["judge_verdicts"] if x is not None)) > 1)

    # Errors can only occur on attempts that actually succeeded, so a
    # marginal error rate is diluted by every failure the judge gets
    # right trivially by saying NO. Report the conditional rates.
    pos = [r for r in usable if r["objective_success"]]
    neg = [r for r in usable if not r["objective_success"]]
    fn = sum(1 for r in pos if not majority(r["judge_verdicts"]))
    fp = sum(1 for r in neg if majority(r["judge_verdicts"]))
    flips = sum(1 for r in usable
                if len(set(x for x in r["judge_verdicts"] if x is not None)) > 1)

    marginal = (fn + fp) / len(usable)
    print(f"marginal error : {fn+fp}/{len(usable)} ({100*marginal:.1f}%) "
          f"— diluted by the failure-heavy base rate, see below")
    print()
    if pos:
        print(f"false NEGATIVE rate : {fn}/{len(pos)} "
              f"({100*fn/len(pos):.1f}%) of correct answers marked wrong")
        print(f"                      -> deflates successes, OVERSTATES cost")
    else:
        print("false NEGATIVE rate : undefined (no successful attempts)")
    if neg:
        print(f"false POSITIVE rate : {fp}/{len(neg)} "
              f"({100*fp/len(neg):.1f}%) of wrong answers marked correct")
        print(f"                      -> inflates successes, UNDERSTATES cost")
    else:
        print("false POSITIVE rate : undefined (no failed attempts)")
    print()
    print(f"judge variance : {flips}/{len(usable)} "
          f"({100*flips/len(usable):.1f}%) flip across repetitions")
    if flips == 0 and manifest and manifest.get("k", 1) > 1:
        t = manifest.get("temperature")
        print(f"  Zero flips at temperature {t}: the judge is deterministic, "
              f"so the\n  k={manifest['k']} repetitions measured nothing. "
              f"Judge VARIANCE is only\n  observable at temperature > 0; "
              f"judge ERROR is the live risk at T=0.")
    if unparseable:
        print(f"unparseable    : {unparseable} (excluded, never guessed)")

    # ---------- 3: does error track the condition? -------------------
    print(f"\n{'cell':8} {'n':>4} {'obj succ':>9} {'judged':>8} "
          f"{'err rate':>9} {'flip rate':>10}")
    print("-" * 54)
    by_cell = defaultdict(list)
    for r in usable:
        by_cell[r["cell"]].append(r)
    err_rates = {}
    for cell in sorted(by_cell):
        rs = by_cell[cell]
        obj = sum(1 for r in rs if r["objective_success"])
        jud = sum(1 for r in rs if majority(r["judge_verdicts"]))
        err = sum(1 for r in rs
                  if majority(r["judge_verdicts"]) != r["objective_success"])
        flip = sum(1 for r in rs
                   if len(set(x for x in r["judge_verdicts"]
                              if x is not None)) > 1)
        err_rates[cell] = err / len(rs)
        print(f"{cell:8} {len(rs):>4} {obj:>9} {jud:>8} "
              f"{100*err/len(rs):>8.1f}% {100*flip/len(rs):>9.1f}%")

    # A raw max-minus-min across eight cells is a weak statistic: it is
    # driven by the two extreme cells and discards the factorial
    # structure. The experiment manipulates S, B and L, so that is what
    # judge error should be tested against — 2 groups of ~n/2 rather
    # than 8 groups of ~n/8. Permutation test per factor, stdlib only.
    print()
    print("does the FALSE-NEGATIVE rate depend on each factor?")
    print(f"  {'factor':8} {'FN at 0':>9} {'FN at 1':>9} {'diff':>7} {'p':>8}"
          f"   (n = successes only)")
    print("  " + "-" * 46)

    rng = random.Random(0)
    confounded = []
    for idx, fac in ((1, "S"), (3, "B"), (5, "L")):
        g0 = [r for r in usable if r["cell"][idx] == "0"]
        g1 = [r for r in usable if r["cell"][idx] == "1"]
        if not g0 or not g1:
            continue
        # Condition on actual success: the false-negative rate is the
        # judge's behaviour. A marginal error rate would partly track
        # each group's success count instead.
        g0 = [r for r in g0 if r["objective_success"]]
        g1 = [r for r in g1 if r["objective_success"]]
        if len(g0) < 3 or len(g1) < 3:
            print(f"  {fac:8} too few successes to test conditionally")
            continue
        def err(rs):
            return sum(1 for r in rs
                       if not majority(r["judge_verdicts"])) / len(rs)
        e0, e1 = err(g0), err(g1)
        obs = abs(e1 - e0)
        labels = [0 if majority(r["judge_verdicts"]) else 1
                  for r in g0 + g1]
        n0, n_perm, hits = len(g0), 5000, 0
        for _ in range(n_perm):
            rng.shuffle(labels)
            a = sum(labels[:n0]) / n0
            b = sum(labels[n0:]) / (len(labels) - n0)
            if abs(b - a) >= obs:
                hits += 1
        pv = (hits + 1) / (n_perm + 1)

        # A bootstrap interval on the difference, because a p-value alone
        # invites exactly the dichotomous call this project argues against.
        boot = []
        for _ in range(2000):
            a = [rng.choice(g0) for _ in g0]
            b = [rng.choice(g1) for _ in g1]
            boot.append(err(b) - err(a))
        boot.sort()
        lo, hi = boot[int(.025*len(boot))], boot[int(.975*len(boot))]

        direction = "worse at 1" if e1 > e0 else "worse at 0"
        print(f"  {fac:8} {100*e0:>8.1f}% {100*e1:>8.1f}% "
              f"{100*obs:>6.1f}% {pv:>8.3f}   "
              f"95% CI [{100*lo:+.0f}, {100*hi:+.0f}] pp, {direction}")
        if pv < 0.10:
            confounded.append((fac, e0, e1, pv, lo, hi))

    n_succ = sum(1 for r in usable if r["objective_success"])
    print(f"\n  (tested on {n_succ} successful attempts — the only ones on")
    print("   which a false negative is possible)")
    if confounded:
        for fac, e0, e1, pv, lo, hi in confounded:
            worse = "compressed" if fac == "S" else f"{fac}=1"
            print(f"\n  {fac}: false negatives run {100*e0:.0f}% -> {100*e1:.0f}% "
                  f"(p={pv:.3f}).")
            print(f"     The judge is systematically worse on one level of a")
            print(f"     factor the experiment manipulates. That is a CONFOUND,")
            print(f"     not noise: it removes successes from one arm faster")
            print(f"     than the other, so it EXAGGERATES OR INVENTS a cost")
            print(f"     difference rather than blurring one.")
        print("\n  Note this is reported as graded evidence, not a verdict.")
        print("  With this many successes the test is underpowered, so an")
        print("  absence of significance is not evidence of absence.")
    else:
        print("\n  No factor shows evidence of dependence at this sample size.")
        print("  Given the low power, read that as inconclusive rather than")
        print("  as a clean bill of health.")

    # ---------- would the conclusion change? -------------------------
    def contrast(use_judge):
        cells = {}
        for cell, rs in by_cell.items():
            tok = [r["tok_total"] for r in rs]
            suc = [int(majority(r["judge_verdicts"]) if use_judge
                       else r["objective_success"]) for r in rs]
            tasks = [r["task_id"] for r in rs]
            if sum(suc) < 2:
                return None
            cells[cell] = (tok, suc, tasks)
        need = ("S0B0L0", "S1B1L0")
        if not all(c in cells for c in need):
            return None
        off, on = cells[need[0]], cells[need[1]]
        d_off, d_on = decompose(off[0], off[1]), decompose(on[0], on[1])
        se_on = cluster_bootstrap_se(*on, n_resamples=args.n_boot)
        return (1 - d_on.T / d_off.T, d_off.T, d_on.T,
                mde_relative(d_on.T, se_on))

    a, b = contrast(False), contrast(True)
    if a and b:
        print(f"\n{'':22} {'RTW':>9} {'T_off':>10} {'T_on':>10} {'MDE':>8}")
        print("-" * 62)
        print(f"{'objective judge':22} {a[0]:>8.1%} {a[1]:>10,.0f} "
              f"{a[2]:>10,.0f} {a[3]:>7.0%}")
        print(f"{'LLM judge':22} {b[0]:>8.1%} {b[1]:>10,.0f} "
              f"{b[2]:>10,.0f} {b[3]:>7.0%}")
        shift = abs(b[0] - a[0])
        print(f"\nheadline contrast moves by {100*shift:.1f} percentage points "
              f"when judged by an LLM.")
        if shift > abs(a[0]) * 0.25:
            print("  *** That is a material change to the reported result.")
    else:
        print("\n(headline contrast not computable — too few successes in a "
              "required cell)")


if __name__ == "__main__":
    main()
