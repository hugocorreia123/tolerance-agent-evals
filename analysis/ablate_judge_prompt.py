#!/usr/bin/env python3
r"""How much of the judge's error is the prompt, not the model?

Stage 1 measured a 73% false-negative rate for a 3B judge and reported
it as a property of weak judges. Inspecting what the judge was actually
shown undermines that: on every successful GSM8K attempt it compared

    Reference answer: 7425.0
    Submitted answer: 7425

because `expected` is stored as a float and `got` is the model's raw
string. A small model asked "is the submitted answer correct?" sees two
strings that differ and says no. That is a presentation defect in the
harness, not a reasoning failure in the model — and it inflates the
false-negative rate by an unknown amount.

This ablation measures how much. Four variants, same attempts, same
model, same temperature:

    baseline   what Stage 1 actually ran
    normalised drop the trailing .0 so both sides read 7425
    tolerant   baseline presentation, prompt explicitly permits
               formatting differences
    bare       no question text at all — pure numeric comparison

If `normalised` or `tolerant` collapses the false-negative rate, then
Stage 1's headline number belongs to the harness rather than to the
model, and the finding must be restated.

    python3 analysis/ablate_judge_prompt.py RESULTS.jsonl SUITE.json \
        --model mlx --n-pos 60 --n-neg 60
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.judge_arm import (MockJudge, MLXJudge, GroqJudge,
                                load_attempts, load_questions)


def fmt(v):
    """Render a reference value the way a person would write it:
    7425.0 -> 7425, but 21.75 -> 21.75."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f == int(f) else str(f)


VARIANTS = {
    "baseline": dict(
        norm=False, question=True,
        template=("You are grading a math answer.\n\nQuestion:\n{question}\n\n"
                  "Reference answer: {expected}\nSubmitted answer: {got}\n\n"
                  "Is the submitted answer correct? Reply with exactly one "
                  "word: YES or NO.")),
    "normalised": dict(
        norm=True, question=True,
        template=("You are grading a math answer.\n\nQuestion:\n{question}\n\n"
                  "Reference answer: {expected}\nSubmitted answer: {got}\n\n"
                  "Is the submitted answer correct? Reply with exactly one "
                  "word: YES or NO.")),
    "tolerant": dict(
        norm=False, question=True,
        template=("You are grading a math answer.\n\nQuestion:\n{question}\n\n"
                  "Reference answer: {expected}\nSubmitted answer: {got}\n\n"
                  "Are these the same number? Formatting differences do not "
                  "matter — 7425.0 and 7425 are the same, as are 1,200 and "
                  "1200.\nReply with exactly one word: YES or NO.")),
    "bare": dict(
        norm=False, question=False,
        template=("Are these two numbers equal in value?\n\n"
                  "A: {expected}\nB: {got}\n\n"
                  "Ignore formatting. Reply with exactly one word: YES or NO.")),
}


class VariantJudge:
    """Wraps a backend so the prompt template can be swapped per variant."""

    def __init__(self, backend, spec):
        self.backend = backend
        self.spec = spec

    def verdict(self, question, expected, got, truth=None):
        e = fmt(expected) if self.spec["norm"] else expected
        prompt = self.spec["template"].format(
            question=question if self.spec["question"] else "", expected=e,
            got=got)
        if isinstance(self.backend, MockJudge):
            return self.backend.verdict(question, e, got, truth)
        return self.backend.complete(prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_file")
    ap.add_argument("suite_file")
    ap.add_argument("--model", default="mock",
                    choices=["mock", "mlx", "groq"])
    ap.add_argument("--variants", default=None,
                    help="comma-separated subset, e.g. baseline,normalised "
                         "— use this to stay inside a hosted quota")
    ap.add_argument("--model-id",
                    default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    ap.add_argument("--n-pos", type=int, default=60,
                    help="successful attempts to sample (false negatives "
                         "are only possible on these)")
    ap.add_argument("--n-neg", type=int, default=60,
                    help="failed attempts to sample (for false positives)")
    args = ap.parse_args()

    attempts = load_attempts(Path(args.results_file))
    questions = load_questions(Path(args.suite_file))
    attempts = [a for a in attempts if a["task_id"] in questions]
    pos = [a for a in attempts if a.get("success")]
    neg = [a for a in attempts if not a.get("success")]
    rng = random.Random(0)
    pos = rng.sample(pos, min(args.n_pos, len(pos)))
    neg = rng.sample(neg, min(args.n_neg, len(neg)))
    sample = pos + neg

    if args.model == "mock":
        backend = MockJudge()
    elif args.model == "mlx":
        backend = MLXJudge(args.model_id, 0.0)
    else:
        gid = (args.model_id if "mlx-community" not in args.model_id
               else "llama-3.3-70b-versatile")
        backend = GroqJudge(gid, 0.0)

    variants = VARIANTS
    if args.variants:
        want = [v.strip() for v in args.variants.split(",")]
        missing = [v for v in want if v not in VARIANTS]
        if missing:
            print(f"unknown variant(s): {missing}")
            sys.exit(1)
        variants = {k: VARIANTS[k] for k in want}
    print(f"sample : {len(pos)} successes + {len(neg)} failures = "
          f"{len(sample)} attempts")
    print(f"judge  : {getattr(backend, 'name', args.model)}")
    print(f"calls  : {len(sample) * len(variants)}\n")

    results = {}
    for vname, spec in variants.items():
        j = VariantJudge(backend, spec)
        fn = fp = unparsed = 0
        for i, a in enumerate(sample, 1):
            q = questions[a["task_id"]]
            lang = "pt" if a["cell"].endswith("L1") else "en"
            v = j.verdict(q[lang] or q["en"], q["expected"], a.get("got"),
                          bool(a.get("success")))
            if v is None:
                unparsed += 1
            elif a.get("success") and not v:
                fn += 1
            elif not a.get("success") and v:
                fp += 1
        results[vname] = (fn, fp, unparsed)
        fn_s = f"{100*fn/len(pos):>5.1f}%" if pos else "  n/a"
        fp_s = f"{100*fp/len(neg):>5.1f}%" if neg else "  n/a"
        print(f"  {vname:11} FN {fn:>3}/{len(pos)} ({fn_s})   "
              f"FP {fp:>3}/{len(neg)} ({fp_s})"
              + (f"   [{unparsed} unparseable]" if unparsed else ""))

    if not pos:
        print("\nno successful attempts in the sample — false negatives "
              "cannot be measured")
        return
    print("\n" + "=" * 62)
    if "baseline" not in results:
        return
    base_fn = results["baseline"][0] / len(pos)
    best = min(results, key=lambda k: results[k][0])
    best_fn = results[best][0] / len(pos)
    print(f"baseline false-negative rate : {100*base_fn:.1f}%")
    print(f"best variant ({best})".ljust(30) + f": {100*best_fn:.1f}%")
    if base_fn > 0:
        share = (base_fn - best_fn) / base_fn
        print(f"\n{100*share:.0f}% of the baseline false-negative rate is "
              f"removed by prompt/presentation")
        print("changes alone — it was never a property of the model.")
        if share > 0.5:
            print("\n*** Stage 1's 73% figure is substantially a harness")
            print("*** defect. The finding must be restated: judge error is")
            print("*** an artefact of how the comparison is posed, and only")
            print("*** the residual belongs to model capability.")


if __name__ == "__main__":
    main()
