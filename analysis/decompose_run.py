#!/usr/bin/env python3
"""Stage 0 — ratio-variance decomposition on an existing Prism results file.

Requires no new inference: every number comes from runs already on disk.

    python analysis/decompose_run.py results/<run>.jsonl

Reports, per experimental cell:
  * T, the all-in tokens per successful task
  * three standard errors — naive, delta-method, cluster-bootstrap
  * how much of the variance each component contributes
  * the minimum detectable effect: the smallest change in cost per
    success this suite could distinguish from noise
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.ratio_variance import (decompose, cluster_bootstrap_se,
                                     mde_relative)


def load(path: Path):
    """Group attempt summaries by cell. Harness errors are excluded —
    they are not experimental observations — but truncations and refusals
    are kept as failures with their tokens intact, which is the whole
    point of an all-in metric."""
    cells = defaultdict(lambda: {"tokens": [], "successes": [], "tasks": []})
    skipped = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("type") != "attempt_summary":
            continue
        if r.get("outcome") == "harness_error":
            skipped += 1
            continue
        c = cells[(r.get("source", "unknown"), r["cell"])]
        c["tokens"].append(r["tok_in_total"] + r["tok_out_total"])
        c["successes"].append(1 if r.get("success") else 0)
        c["tasks"].append(r["task_id"])
    return cells, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_file")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--power", type=float, default=0.80)
    args = ap.parse_args()

    cells, skipped = load(Path(args.results_file))
    if not cells:
        print("no usable attempt summaries found")
        sys.exit(1)

    sources = sorted({src for src, _ in cells})
    print(f"file    : {args.results_file}")
    print(f"sources : {len(sources)} ({', '.join(sources)})"
          + (f"   ({skipped} harness errors excluded)" if skipped else ""))
    if len(sources) > 1:
        print()
        print("*** Reporting PER SOURCE. Pooling them would mix populations")
        print("*** with different token scales, and the variance would then")
        print("*** be dominated by which source was sampled rather than by")
        print("*** the experimental factors.")

    for src in sources:
        rows = []
        print(f"\n{'='*84}\nsource: {src}\n{'='*84}")
        print(f"{'cell':8} {'n':>4} {'succ':>5} {'T':>10} "
              f"{'naive SE':>9} {'delta SE':>9} {'boot SE':>9} "
              f"{'narrow by':>10} {'MDE':>7}")
        print("-" * 84)
        for (s_, cell) in sorted(cells):
            if s_ != src:
                continue
            c = cells[(s_, cell)]
            n, nsucc = len(c["tokens"]), sum(c["successes"])
            if nsucc < 2:
                print(f"{cell:8} {n:>4} {nsucc:>5} "
                      f"{'-':>10}  too few successes for a ratio estimate")
                continue
            d = decompose(c["tokens"], c["successes"])
            boot = cluster_bootstrap_se(c["tokens"], c["successes"],
                                        c["tasks"], n_resamples=args.n_boot)
            narrow = 100 * (1 - d.naive_se / boot) if boot > 0 else float("nan")
            mde_rel = mde_relative(d.T, boot, power=args.power)
            print(f"{cell:8} {n:>4} {nsucc:>5} {d.T:>10,.0f} "
                  f"{d.naive_se:>9,.0f} {d.delta_se:>9,.0f} {boot:>9,.0f} "
                  f"{narrow:>9.0f}% {mde_rel:>6.0%}")
            rows.append((cell, d, boot, mde_rel))

        if not rows:
            continue
        print()
        print("variance decomposition (delta method, share of total)")
        print(f"{'cell':8} {'tokens':>9} {'covariance':>12} {'success rate':>14}")
        print("-" * 46)
        for cell, d, _, _ in rows:
            print(f"{cell:8} {100*d.share_tokens:>8.1f}% "
                  f"{100*d.share_covariance:>11.1f}% "
                  f"{100*d.share_success:>13.1f}%")

        worst = max(rows, key=lambda r: r[3])
        best = min(rows, key=lambda r: r[3])
        mean_narrow = sum(100 * (1 - d.naive_se / b)
                          for _, d, b, _ in rows) / len(rows)
        mean_cov = sum(100 * d.share_covariance for _, d, _, _ in rows) / len(rows)
        mean_suc = sum(100 * d.share_success for _, d, _, _ in rows) / len(rows)
        print()
        print(f"  naive interval is on average {mean_narrow:.0f}% too narrow")
        print(f"  omitted terms: covariance {mean_cov:+.0f}%, "
              f"success-rate {mean_suc:.0f}% of total variance")
        print(f"  minimum detectable change in cost per success: "
              f"{best[3]:.0%} ({best[0]}) to {worst[3]:.0%} ({worst[0]})")


if __name__ == "__main__":
    main()
