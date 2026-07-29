# Tolerance — Findings

## Can your agent evaluation detect the improvement you are claiming?

Hugo Correia · July 2026

---

## Abstract

Agentic evaluations increasingly report a **cost-efficiency** headline: tokens, or
dollars, per successfully completed task. That quantity is a **ratio estimator** —
both its numerator and its denominator are random — and it does not behave like a
mean. Treating it as one produces confidence intervals that are far too narrow and
power calculations that are far too optimistic.

This report quantifies the error. In simulation against a known truth, a nominal 95%
interval computed the usual way achieves **39% actual coverage**. Applied to a real
848-run agentic study, the naive interval is **71–81% too narrow**, and the dominant
variance component — contributing **81% of total variance on single-turn tasks and
91% on multi-turn tasks** — is the one the naive treatment omits entirely.

The practical consequence is a **minimum detectable effect of 121–222%** for that
study's single-turn suite. Applied to the study's own published headline, the observed
effect of −109.7% sits *below* its own detection threshold. That study had already
conceded in writing that it was "underpowered for its own primary metric"; this work
converts the concession into a number.

A design sweep adds a budget rule: in a within-task paired design the standard error
is **flat** in replicates per task, while an independent design's standard error
**nearly doubles** over the same range — so paired designs can trade tasks for
replicates freely, and independent designs must maximise task count.

**Stage 1** removes the assumption that judging is objective. Re-scoring the same 640
attempts with an LLM judge shows judge *variance* is exactly **zero** at temperature 0,
while judge *error* removes **70% of all successes** (73% false-negative rate, 0.8%
false-positive). The point estimate barely moves — 13.4 percentage points — but the
**minimum detectable effect exactly doubles**, from 209% to 421%, matching the √n law.
An LLM judge does not change what you conclude; it halves your ability to conclude
anything.

---

## 1. The question

If an evaluation reports that intervention X reduced cost per successful task by 15%,
a reader needs to know one thing before believing it: **could this evaluation have
detected 15%?**

Answering that requires the sampling distribution of the metric. For accuracy — a
simple proportion — this is well understood, and recent work has brought error bars
and power analysis to single-turn benchmarks. For **cost per success** it has not been
done, and the metric is not a proportion. It is a ratio of two random quantities, and
that changes its variance structure fundamentally.

---

## 2. Theory: a ratio is not a mean

### 2.1 The metric

For a set of *n* attempts, let *tᵢ* be the total tokens spent on attempt *i*
(successes and failures alike) and *sᵢ* ∈ {0,1} its outcome. The reported metric is

```
T  =  Σ tᵢ / Σ sᵢ
```

Every failed attempt contributes to the numerator and nothing to the denominator,
which is the entire point: a failure is billed at full price.

### 2.2 Why the usual treatment is wrong

The intuitive approach treats the success count as a fixed divisor and propagates only
the variability of the token total. That is equivalent to assuming the denominator is
known in advance. It is not — it is a random variable, and often a small one.

By the delta method, for T = ȳ/x̄ with y = tokens and x = success indicator:

```
Var(T)  ≈   1/(n·x̄²) · [  S²ᵧ   −   2·T·Sₓᵧ   +   T²·S²ₓ  ]
                          ─────      ────────       ───────
                          tokens     covariance    success rate
```

Three terms, of which the naive treatment retains only the first.

**The success-rate term** (T²·S²ₓ) is strictly positive and grows with T². Since T is
large whenever the success rate is low, this term dominates precisely in the regime
where agentic evaluations operate.

**The covariance term** is signed, and its sign is informative. When failures cost
*more* than successes — the normal case for agents, because a failing attempt burns
its full budget through retries and truncation — Sₓᵧ is negative, so −2·T·Sₓᵧ is
**positive** and inflates variance further.

Both omissions therefore push the same way. A naive interval is too narrow, never too
wide.

### 2.3 A third layer: clustering

Attempts are nested within tasks, and tasks differ enormously in difficulty and cost.
Resampling individual attempts would treat correlated observations as independent.
All intervals reported here come from a **cluster bootstrap that resamples task
identifiers**, never rows.

---

## 3. Method

Three estimators are compared throughout:

| Estimator | Ratio-aware | Cluster-aware |
|---|---|---|
| **naive** | no | no |
| **delta** | yes | no |
| **cluster bootstrap** | yes | yes |

**Validation before application.** No estimator is trusted to report on itself. Ground
truth is established by simulating many datasets from a generating process whose true
value is known, then measuring what fraction of nominal 95% intervals actually contain
it. Only estimators that survive that test are applied to real data.

**Real data.** The empirical work uses the complete results of *Prism*, a
pre-registered 2×2×2 factorial study of LLM token-reduction techniques: 848 runs of
Qwen2.5-3B across 53 tasks — 40 GSM8K-hard items wrapped with calculator and lookup
tools, and 13 BFCL-v4 multi-turn agent tasks. Judgement is objective in both
(`numeric_exact`; final-state comparison against a vendored simulator), so **judge
variance is zero by construction** — which isolates the components under study, and
also bounds what these results generalise to (§6).

Everything in this report runs on a laptop with **no third-party dependencies**.

---

## 4. Stage 0 — the variance of a ratio

### 4.1 A nominal 95% interval achieves 39% coverage

Over 300 simulated datasets with a known population value:

| Estimator | Nominal | **Actual coverage** |
|---|---|---|
| Naive | 95% | **39.0%** |
| Delta method | 95% | 93.7% |
| Cluster bootstrap | 95% | 94.3% |

A naive interval on cost-per-success is wrong more often than it is right. Both
corrected estimators are calibrated.

A secondary observation from the same simulations: across 60 independent datasets the
bootstrap standard error averaged 105% of truth — essentially unbiased — but ranged
from 352 to 938. **The uncertainty estimate is itself unstable** at 40 tasks, which is
a second-order problem worth naming.

### 4.2 On real data, the omitted terms are the largest ones

Variance decomposition across all cells of the 848-run study:

| Suite | Token variance | Covariance | **Success-rate** | Naive interval too narrow by |
|---|---|---|---|---|
| GSM8K (single-turn) | 6–17% | +7% avg | **81% avg** | **81%** |
| BFCL (multi-turn) | 1–5% | +7% avg | **91% avg** | **71%** |

The component practitioners actually model — token variance — accounts for as little
as 1% of the total. The component nobody models accounts for up to 94%.

The covariance term is **positive in every cell of both suites**, confirming on real
data the mechanism predicted in §2.2: failures cost more than successes, so the term
inflates rather than cancels.

### 4.3 Minimum detectable effect: 121–222%

| Suite | MDE range (80% power) |
|---|---|
| GSM8K | 121% – 222% |
| BFCL | 116% – 403% |

An improvement smaller than this cannot be distinguished from noise by this suite. For
context, published agentic cost-efficiency improvements are routinely reported in the
range of 10–40%.

One BFCL cell returned **no estimate at all** — a single success across 26 attempts is
not enough to form a ratio. That refusal is a result: below roughly two or three
successes per cell, the metric is not estimable, and reporting a point value for it is
not meaningful.

### 4.4 The method re-audits the study it came from

Applying this to *Prism*'s own headline:

| | Value |
|---|---|
| Baseline cell (S0B0L0) | T = 4,305 tokens per success |
| Fully-optimised cell (S1B1L0) | T = 9,026 tokens per success |
| Implied RTW | **−109.7%** — matching Prism's published figure exactly |
| MDE for that cell | **121%** |

**The observed effect sits below the detection threshold.** Prism reported a BCa
interval of (−336.6%, −3.2%) that excluded zero — but only just, and BCa is asymmetric,
so it can clear zero where a symmetric interval would not.

Prism's own limitations section states that the study is "underpowered for its own
primary metric," having run its power analysis on token counts rather than on
tokens-per-success. This work quantifies exactly how much.

### 4.5 A budget rule from the design sweep

Comparing two designs at **equal attempt budget** — paired (*n* tasks, each run *R*
times in both cells) versus independent (*2n* tasks, half in each cell) — using true
standard errors from simulation:

| Replicates per task per cell | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| **SE, paired design** | 0.398 | 0.389 | 0.388 | 0.383 | 0.401 |
| **SE, independent design** | 0.431 | 0.483 | 0.550 | 0.672 | 0.855 |
| Pairing advantage | 1.08× | 1.24× | 1.42× | 1.75× | 2.13× |

**The paired standard error is flat in R; the independent one nearly doubles.** The
pairing advantage grows not because pairing improves but because the independent
design degrades as tasks are traded for replicates.

The rule that follows:

> In a **within-task** design, how you split budget between tasks and replicates
> barely matters. In an **independent** design, maximise the number of tasks.

The pattern holds across three regimes (task difficulty expressed mainly through token
scale, mainly through success rate, and through both).

A note on how this result arrived: an earlier version of this analysis concluded that
pairing bought nothing at all. That comparison was between two *estimators* on one
dataset, which is the wrong comparison — the practitioner's question is between two
*designs* at fixed budget. The corrected experiment reversed the finding.

### 4.6 The error shrinks with success rate but never disappears

Section 4.1 measured coverage at one operating point. Because the dominant term scales
as `T²·p(1−p)` and `T ~ 1/p`, both factors shrink as evaluations get easier — so the
naive omission should matter less at high success rates. Sweeping the success rate
locates how much less:

| Success rate | Naive coverage | Delta coverage | Success-rate share of variance | Correct SE vs naive | MDE |
|---|---|---|---|---|---|
| 10% | **15%** | 93% | 88% | 10.40× | 128% |
| 20% | 22% | 92% | 78% | 6.47× | 80% |
| 30% | 33% | 94% | 69% | 4.70× | 63% |
| 50% | 39% | 92% | 51% | 3.08× | 46% |
| 70% | 58% | 93% | 36% | 2.27× | 37% |
| 85% | 64% | 89% | 26% | 1.86× | 28% |
| 95% | **68%** | 88% | 20% | 1.59× | 21% |

**There is no threshold above which the naive interval becomes acceptable.** It
improves monotonically and still reaches only 68% coverage at a 95% success rate — the
easiest regime that exists. The correction is not situational.

The second consequence is larger. **The minimum detectable effect falls from 128% to
21% purely as a function of success rate.** At 95% success an evaluation can resolve
the 10–40% improvements practitioners actually report; at 10% success it cannot resolve
anything short of a doubling.

> Difficulty and precision are coupled through the same denominator. Making a benchmark
> harder does not merely lower the scores — it destroys the ability to measure changes
> to them.

This also explains why Stage 0's real-data MDEs are so large: the study analysed runs
at 16–31% success rates, which sits at the punishing end of this curve.

### 4.7 A second corpus confirms the mechanism

§4.6 is simulation. The same study provides an empirical test at the opposite end of
the curve: its M2 arm ran the identical suite on Llama-3.3-70B at a **94% success rate**
against M1's 31%.

| | M1 · Qwen2.5-3B | M2 · Llama-3.3-70B |
|---|---|---|
| Success rate | ~31% | ~94% |
| Success-rate share of variance | **81%** | **16%** |
| Covariance share | +7% | +25% |
| Token share | 6–17% | 31–100% |
| Naive interval too narrow by | **81%** | **44%** |
| MDE (as measured) | 121–222% | 87–197% |

The simulation predicted a success-rate share of 69% at p = 30% and 20% at p = 95%. The
observed values are 81% and 16%. The mechanism transfers.

**The composition inverts, but the error does not disappear.** At low success rates the
success-rate term dominates; at high rates token variance and covariance do. What the
naive estimator omits predicts how wrong it is almost exactly:

| | Omitted variance | Interval too narrow by |
|---|---|---|
| M1 | 7% + 81% = **88%** | 81% |
| M2 | 25% + 16% = **41%** | 44% |

**The MDE comparison is confounded by sample size** and should not be read directly: M2
has 13 attempts per cell against M1's 80, a √(80/13) = 2.48× penalty. Adjusted to equal
n, M2's MDE would be roughly **35–79%** against M1's 121–222% — so a high success rate
roughly halves to thirds the detectable effect, consistent with §4.6.

**A degenerate cell completes the argument.** In M2's S1B1L1 every attempt succeeded
(12/12), so the success-rate variance is exactly zero and the covariance vanishes: the
ratio estimator collapses to a simple mean and the naive standard error equals the delta
standard error exactly (303 = 303). Yet the cluster bootstrap still returns 407 — **26%
wider**. Even when both ratio corrections vanish, clustering remains. There are three
things the naive treatment ignores, and at no operating point are all three absent.

---

## 5. Stage 1 — what an LLM judge costs

Stage 0's stated limitation was that both suites judge objectively, so judge variance
is zero by construction and the results might not transfer to LLM-as-judge evaluation —
which is how most agent systems are scored.

**The trajectories were already spent.** Prism records `expected`, `got` and the
objective verdict for every attempt, so testing this required no agent re-runs, only
re-judging. All 640 GSM8K attempts were re-scored by a local Qwen2.5-3B judge at
temperature 0.

### 5.1 Judge variance is zero; judge error is not

| Quantity | Result |
|---|---|
| Judge variance (run-to-run flips) | **0 / 640 = 0.0%** |
| False negatives (correct answers marked wrong) | **111 / 152 = 73.0%** |
| False positives (wrong answers marked correct) | 4 / 488 = 0.8% |

At temperature 0 the judge is deterministic, so repeated judging measures nothing. An
initial run used k=3 repetitions; they were wasted compute. **Judge variance is only
observable at temperature > 0, and practitioners run judges at temperature 0.** The
live risk is error, and the two are routinely conflated.

The error is severely asymmetric: the judge marks nearly three-quarters of correct
answers wrong, and almost never does the reverse.

### 5.2 The cost is paid in power, not in the estimate

Successes fall from **152 to 45** — 70% removed. The headline contrast:

| | RTW | T_off | T_on | **MDE** |
|---|---|---|---|---|
| Objective judge | −109.7% | 4,305 | 9,026 | **209%** |
| LLM judge | −96.3% | 11,957 | 23,469 | **421%** |

**The point estimate moves 13.4 percentage points. The MDE exactly doubles (2.01×).**

That is not coincidence and it is not bias. Because the judge's error is roughly uniform
across cells, it largely cancels in a ratio of ratios. What it destroys is effective
sample size: successes fell 3.38×, and MDE scales as 1/√n, predicting 1.84× against
2.01× observed.

> An LLM judge does not change what you conclude. It halves your ability to conclude
> anything.

This follows directly from Stage 0: the success count is the dominant variance term, so
anything that removes successes inflates the MDE.

### 5.3 Does judge error track the experimental condition?

If judge error correlated with a manipulated factor it would be a **confound** rather
than noise — creating or erasing effects instead of blurring them. Tested per factor,
conditional on actual success (errors are only possible where a success occurred):

| Factor | FN rate at 0 | FN rate at 1 | Difference | p | 95% CI |
|---|---|---|---|---|---|
| S (schema) | 67.4% | 80.3% | 12.9 pp | 0.101 | [−1, +27] pp |
| B (budget) | 75.0% | 70.8% | 4.2 pp | 0.587 | [−19, +10] pp |
| L (language) | 72.6% | 73.5% | 0.9 pp | 1.000 | [−14, +14] pp |

Suggestive evidence for S; not established. B and L show nothing.

**This estimate is itself a lesson.** A 200-attempt subsample run first gave S a
difference of **27.4 pp at p = 0.056** — an apparently strong effect that shrank to 12.9
pp with an interval spanning zero once all 640 attempts were used. That is the classic
underpowered overestimate, occurring in a study whose entire subject is underpowered
estimates. It is reported here rather than quietly replaced.

---

## 6. What to do with this

**Report a cluster-bootstrap interval, not a naive one.** The correction is cheap and
the error it removes is large.

**Check your MDE before claiming an improvement.** If your suite cannot resolve 100%,
a claimed 20% saving is not evidence of anything.

**Watch the denominator, not the numerator.** Effort spent reducing token-measurement
noise addresses 1–17% of the variance. Effort spent raising the success count, or
adding tasks, addresses the rest.

**Below ~3 successes per cell, do not report the metric.** It is not estimable, and a
point estimate implies a precision that does not exist.

**Prefer a within-task design** — and if you have one, allocate budget between tasks
and replicates however is convenient.

**If you score with an LLM judge, expect to lose most of your statistical power.** A
73% false-negative rate doubled the MDE here. Budget for it, or use an objective judge
wherever the task admits one.

**Do not repeat-run a judge at temperature 0.** It is deterministic; the repetitions
measure nothing.

---

## 7. Limits

**The judge arm uses one judge and one prompt.** Judge behaviour is prompt-sensitive
and model-sensitive; no ablation was run. The 73% false-negative rate is a property of
this 3B judge with this prompt, not of LLM judging in general. A stronger judge would
presumably err less; whether it would err *symmetrically* is untested, and the
asymmetry is what drives the power loss.

**Low power in §5.3.** With 152 successes the factor tests could not resolve differences
below roughly 25 pp. Absence of evidence there is not evidence of absence.

**One study's data, one model family.** The empirical section analyses a single
848-run corpus on Qwen2.5-3B. The direction and mechanism should generalise — they
follow from the algebra in §2 — but the specific magnitudes are properties of this
suite and model.

**The design sweep is simulation.** §4.5 rests on a generating process chosen to
resemble agentic evaluation, not on empirical paired/independent runs.

**MDE is computed for a two-arm comparison** at 80% power using a normal
approximation. At the very low success counts seen in the BFCL cells, that
approximation is itself strained — a limitation that argues for the same conclusion as
§4.3.

**No empirical replicate sweep.** §4.5 is simulated; a paired-versus-independent
comparison on real runs has not been done.

---

## 8. Reproducing

```bash
# validate the estimators against known ground truth
python3 analysis/validate_ratio_variance.py

# decompose any Prism-format results file
python3 analysis/decompose_run.py results/<run>.jsonl

# paired vs independent design comparison
python3 analysis/design_sweep.py

# judge arm (the only part needing a model backend)
python3 analysis/judge_arm.py RESULTS.jsonl SUITE.json --model mlx --k 1
python3 analysis/judge_analysis.py results/judge/<file>.jsonl
```

No third-party packages are required for the statistics — they use only the Python
standard library, so the analysis runs anywhere Python does. Only the judge arm needs a
model backend (`mlx-lm`, or any substitute). All simulations are
deterministically seeded and reproduce byte-identically.

---

## Appendix A — Full cell tables

### A.1 GSM8K, single-turn with tools (80 attempts per cell)

| Cell | Successes | T | naive SE | delta SE | bootstrap SE | Too narrow by | MDE |
|---|---|---|---|---|---|---|---|
| S0B0L0 | 25 | 4,305 | 330 | 799 | 1,318 | 75% | 121% |
| S0B0L1 | 19 | 5,112 | 274 | 1,082 | 1,860 | 85% | 144% |
| S0B1L0 | 24 | 4,723 | 410 | 991 | 1,719 | 76% | 144% |
| S0B1L1 | 18 | 5,676 | 379 | 1,288 | 2,331 | 84% | 163% |
| S1B0L0 | 22 | 5,315 | 476 | 1,156 | 1,782 | 73% | 133% |
| S1B0L1 | 14 | 6,363 | 377 | 1,555 | 3,540 | 89% | 220% |
| S1B1L0 | 13 | 9,026 | 810 | 2,566 | 5,054 | 84% | 222% |
| S1B1L1 | 17 | 6,237 | 521 | 1,550 | 3,284 | 84% | 209% |

Variance shares:

| Cell | Tokens | Covariance | Success rate |
|---|---|---|---|
| S0B0L0 | 17.0% | +2.1% | 80.8% |
| S0B0L1 | 6.4% | +2.8% | 90.8% |
| S0B1L0 | 17.1% | +15.8% | 67.1% |
| S0B1L1 | 8.7% | +6.6% | 84.7% |
| S1B0L0 | 17.0% | +12.5% | 70.5% |
| S1B0L1 | 5.9% | −5.8% | 99.9% |
| S1B1L0 | 10.0% | +9.3% | 80.7% |
| S1B1L1 | 11.3% | +12.7% | 76.0% |

### A.2 BFCL-v4, multi-turn agent tasks (26 attempts per cell)

| Cell | Successes | T | naive SE | delta SE | bootstrap SE | Too narrow by | MDE |
|---|---|---|---|---|---|---|---|
| S0B0L0 | 4 | 609,121 | 47,928 | 302,699 | 618,894 | 92% | 403% |
| S0B0L1 | 2 | 962,727 | 110,285 | 693,713 | 299,926 | 63% | 123% |
| S0B1L0 | 2 | 1,046,154 | 81,923 | 758,640 | 306,397 | 73% | 116% |
| S0B1L1 | 2 | 716,083 | 65,114 | 510,312 | 209,105 | 69% | 116% |
| S1B0L0 | 3 | 618,572 | 65,674 | 354,611 | 522,308 | 87% | 335% |
| S1B0L1 | 2 | 792,301 | 124,872 | 588,250 | 246,076 | 49% | 123% |
| S1B1L0 | 1 | — | — | — | — | — | not estimable |
| S1B1L1 | 2 | 685,812 | 78,769 | 507,052 | 213,886 | 63% | 124% |

Success-rate share of variance ranges from 87.1% to 94.5% across these cells.

### A.3 Judge arm, per cell (80 attempts per cell)

| Cell | Objective successes | Judged successes | Marginal error rate |
|---|---|---|---|
| S0B0L0 | 25 | 9 | 20.0% |
| S0B0L1 | 19 | 5 | 17.5% |
| S0B1L0 | 24 | 8 | 22.5% |
| S0B1L1 | 18 | 8 | 15.0% |
| S1B0L0 | 22 | 4 | 22.5% |
| S1B0L1 | 14 | 2 | 15.0% |
| S1B1L0 | 13 | 5 | 15.0% |
| S1B1L1 | 17 | 4 | 16.2% |
| **Total** | **152** | **45** | 18.0% |

The marginal error rate is shown for completeness but is misleading: errors are only
possible on attempts that actually succeeded, so it is diluted by the 488 failures the
judge classifies correctly by default. The conditional rates in §5.1 are the
informative ones.

---

## Appendix B — Data

The 848-run corpus analysed here is published in full at
[prism-token-taxes](https://github.com/hugocorreia123/prism-token-taxes)
(`results/r20260725_170738_270d87.jsonl`), append-only with a hash-pinned manifest.
Every number in this report can be regenerated from that file with the commands in §7.
