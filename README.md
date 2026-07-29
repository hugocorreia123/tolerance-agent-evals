# Tolerance

**Your evaluation says the change made things 15% cheaper. Could it have detected 15%?**

Usually not. On a real 848-run agentic study, the smallest cost change that could be
distinguished from noise was **121%** — while improvements in this field are routinely
reported at 10–40%.

This repository contains the method, the validation, and the measurements.

---

## The short version

Teams ship prompt changes, model swaps and optimisations on the strength of an
evaluation delta: *this version costs less per completed task, so we shipped it.*

That decision needs one number nobody computes — the **resolution** of the evaluation.
A bathroom scale marked only in 5 kg increments cannot support a claim that you lost
2 kg. It is not that the scale is lying; it is that the claim is finer than the
instrument.

Agent evaluations have a resolution too, and it turns out to be much coarser than
people assume. The reason is specific and fixable.

---

## Why cost-per-success is harder to measure than it looks

The headline metric of most agentic evaluations is some form of

```
cost per successful task  =  everything you spent  /  tasks that actually finished
```

Failures are in the numerator and not the denominator, which is correct — a failed run
is billed at full price.

But it makes the metric a **ratio of two uncertain things**. The tokens vary. The
number of successes also varies, and varies *more*, because success is a coin-flip per
attempt rather than a smooth quantity. Standard practice treats the success count as if
it were fixed and known, and propagates only the token variability.

That single simplification is the whole problem:

| | Share of the real uncertainty |
|---|---|
| Token variability — **what people measure** | 1–17% |
| Success-rate variability — **what people ignore** | 67–99% |

A confidence interval built the usual way is **71–81% too narrow** on real data. In
simulation against a known answer, a nominal "95% confidence" interval contained the
truth **39% of the time**.

---

## What was found

**1. A 95% interval that is right 39% of the time.**
Validated against 300 simulated datasets with a known population value. The
ratio-aware estimators land at 93.7% and 94.3%; the naive one at 39.0%.

**2. The ignored term is the dominant one.**
Across every cell of a real 848-run study, success-rate variability accounts for 81% of
total variance on single-turn tasks and 91% on multi-turn ones. The term practitioners
actually model accounts for as little as 1%.

**3. The method indicts the study it came from.**
Applied to *Prism* — a published factorial study of token-reduction techniques — it
reproduces that study's headline exactly (−109.7%) and then shows the effect sits
**below its own detection threshold** of 121%. Prism's limitations section had already
conceded it was "underpowered for its own primary metric." This puts a number on it.

**4. An LLM judge halves your statistical power without changing your answer.**
Re-scoring all 640 attempts with an LLM judge: run-to-run judge *variance* is exactly
zero at temperature 0, but judge *error* marks 73% of correct answers wrong and removes
70% of all successes. The point estimate moves 13 percentage points; the minimum
detectable effect **exactly doubles**, 209% → 421%, matching the √n law.

**5. The error never goes away, and difficulty costs precision.**
Sweeping the success rate: naive coverage climbs from 15% to 68% as evaluations get
easier, but **never reaches adequacy** — there is no threshold above which the shortcut
is safe. Meanwhile the minimum detectable effect falls from **128% at a 10% success rate
to 21% at 95%**. A harder benchmark does not just score lower; it loses the ability to
measure changes to itself.

**6. A budget rule for designing evaluations.**
At fixed budget, a within-task paired design's standard error is flat in replicates per
task, while an independent design's nearly doubles. So paired designs can trade tasks
for replicates freely; independent designs must maximise task count.

---

## What to do about it

- **Report a cluster-bootstrap interval**, not a naive one. Cheap correction, large error removed.
- **Check your MDE before claiming an improvement.** If the suite cannot resolve 100%, a claimed 20% saving is not evidence.
- **Watch the denominator.** Effort spent on token-measurement precision addresses 1–17% of the variance. Raising the success count, or adding tasks, addresses the rest.
- **Below ~3 successes per cell, do not report the metric.** It is not estimable, and a point value implies precision that does not exist.
- **If you need to detect small improvements, you need a high success rate.** Difficulty and precision are coupled through the same denominator.
- **If you use an LLM judge, budget for the power loss** — or use an objective judge wherever the task admits one.
- **Never repeat-run a judge at temperature 0.** It is deterministic; the repetitions measure nothing.

---

## The method

For *n* attempts with tokens *tᵢ* and outcomes *sᵢ* ∈ {0,1}, the metric is
`T = Σtᵢ / Σsᵢ`. By the delta method:

```
Var(T)  ≈   1/(n·x̄²) · [  S²ᵧ   −   2·T·Sₓᵧ   +   T²·S²ₓ  ]
                          tokens   covariance   success rate
```

The naive treatment keeps only the first term. The success-rate term is strictly
positive and scales with T², so it dominates whenever success rates are low. The
covariance term is signed — and when failures cost *more* than successes, which is the
normal case for agents, it is positive too. Both omissions push the same way, so a
naive interval is too narrow and never too wide.

Every interval reported here comes from a **cluster bootstrap that resamples task
identifiers**, never individual rows, because attempts within a task are correlated.

Three estimators are compared throughout — naive, delta-method, and cluster bootstrap —
and none is trusted to report on itself: each is validated against simulated data whose
true value is known before being applied to anything real.

---

## Reproducing

```bash
# validate the estimators against known ground truth
python3 analysis/validate_ratio_variance.py

# decompose any Prism-format results file
python3 analysis/decompose_run.py results/<run>.jsonl

# paired vs independent design comparison
python3 analysis/design_sweep.py

# how naive error depends on the success rate
python3 analysis/success_rate_sweep.py

# judge arm — the only part that needs a model backend
python3 analysis/judge_arm.py RESULTS.jsonl SUITE.json --model mlx --k 1
python3 analysis/judge_analysis.py results/judge/<file>.jsonl
```

**The statistics have no third-party dependencies** — Python standard library only, so
they run anywhere Python does. Only the judge arm needs `mlx-lm` or an equivalent
backend. All simulations are deterministically seeded and reproduce byte-identically.

---

## Layout

```
FINDINGS.md              the full report: theory, validation, results, limits
analysis/ratio_variance.py         delta-method decomposition, cluster bootstrap, MDE
analysis/validate_ratio_variance.py  coverage validation against known truth
analysis/decompose_run.py          CLI: decompose a real results file
analysis/design_sweep.py           paired vs independent design comparison
analysis/success_rate_sweep.py     how the error depends on the success rate
analysis/judge_arm.py              re-judge existing runs with an LLM
analysis/judge_analysis.py         judge error, variance, and factor dependence
```

The corpus analysed is published in full at
[prism-token-taxes](https://github.com/hugocorreia123/prism-token-taxes) — append-only,
with hash-pinned manifests. Every number in `FINDINGS.md` regenerates from it.

---

## Two corrections worth reading

Both are in `FINDINGS.md` rather than quietly fixed, because how a result changed is
part of the result.

**Pairing.** An earlier analysis concluded that within-task pairing bought nothing. That
compared two *estimators* on one dataset — the wrong comparison. The practitioner's
question is between two *designs* at fixed budget, and the corrected experiment
reversed the finding.

**An underpowered overestimate, in a study about underpowered estimates.** A
200-attempt subsample suggested the judge's error rate depended strongly on the schema
condition: a 27 percentage-point gap at p = 0.056. Running all 640 attempts shrank it to
13 points with an interval spanning zero. The larger sample corrected the smaller one,
which is exactly what this project argues will happen — and it happened here.
