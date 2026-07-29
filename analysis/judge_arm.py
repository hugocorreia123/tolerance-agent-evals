#!/usr/bin/env python3
r"""Re-judge an existing agentic run with an LLM judge.

No agents are re-run. Prism records `expected` and `got` for every
attempt, so the trajectory is already spent — only the JUDGING is
repeated. That makes the judge arm cheap enough to run locally.

Two quantities are measured, and they are different things:

  judge ERROR    disagreement with the objective ground-truth judge.
                 Biases the success COUNT, therefore biases T.
  judge VARIANCE run-to-run disagreement on identical input. Adds
                 noise to T. At temperature 0 this is ~0, which is
                 exactly why error matters more than practitioners
                 assume.

The question this is built to answer is sharper than either:

  Does judge error CORRELATE WITH THE EXPERIMENTAL CONDITION?

If it does, the judge manufactures or erases effects rather than
merely blurring them — a confound, not noise.

    python analysis/judge_arm.py RESULTS.jsonl SUITE.json --model mlx
    python analysis/judge_arm.py RESULTS.jsonl SUITE.json --model mock
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

JUDGE_PROMPT = """You are grading a math answer.

Question:
{question}

Reference answer: {expected}
Submitted answer: {got}

Is the submitted answer correct? Reply with exactly one word: YES or NO."""


class MockJudge:
    """Deterministic-ish stand-in for wiring tests. Agrees with ground
    truth most of the time and flips with a fixed probability, so the
    analysis path can be exercised without a model."""
    name = "mock-judge"

    def __init__(self, flip_rate=0.12, seed=0):
        self.flip_rate = flip_rate
        self.rng = random.Random(seed)

    def verdict(self, question, expected, got, truth):
        return (not truth) if self.rng.random() < self.flip_rate else truth


class MLXJudge:
    """Local judge via mlx-lm. Short prompt, one-token answer."""

    def __init__(self, model_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
                 temperature=0.0):
        from mlx_lm import load, generate
        self.model, self.tokenizer = load(model_id)
        self._generate = generate
        self.name = model_id
        self.temperature = temperature

    def verdict(self, question, expected, got, truth=None):
        prompt = JUDGE_PROMPT.format(question=question, expected=expected,
                                     got=got)
        msgs = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)
        out = self._generate(self.model, self.tokenizer, prompt=text,
                             max_tokens=6, verbose=False)
        up = out.strip().upper()
        if up.startswith("YES"):
            return True
        if up.startswith("NO"):
            return False
        return None          # unparseable — recorded, never guessed


def load_attempts(path: Path):
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("type") != "attempt_summary":
            continue
        if r.get("outcome") == "harness_error":
            continue
        if r.get("source") and "gsm8k" not in r["source"]:
            continue          # judge prompt is written for the math suite
        out.append(r)
    return out


def load_questions(path: Path):
    d = json.loads(path.read_text())
    q = {}
    for t in d["tasks"]:
        q[t["id"]] = {"en": t["content"]["en"], "pt": t["content"].get("pt"),
                      "expected": t["expected"]}
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_file")
    ap.add_argument("suite_file")
    ap.add_argument("--model", default="mock", choices=["mock", "mlx"])
    ap.add_argument("--model-id",
                    default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    ap.add_argument("--k", type=int, default=3,
                    help="judge repetitions per attempt (>1 measures "
                         "judge variance; at temperature 0 expect ~0)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="subsample this many attempts (evenly across cells)")
    ap.add_argument("--out", default="results/judge")
    args = ap.parse_args()

    attempts = load_attempts(Path(args.results_file))
    questions = load_questions(Path(args.suite_file))
    attempts = [a for a in attempts if a["task_id"] in questions]
    if not attempts:
        print("no gsm8k attempts found that match the suite")
        sys.exit(1)

    if args.limit:
        by_cell = {}
        for a in attempts:
            by_cell.setdefault(a["cell"], []).append(a)
        per = max(1, args.limit // len(by_cell))
        rng = random.Random(0)
        attempts = [x for v in by_cell.values()
                    for x in rng.sample(v, min(per, len(v)))]

    judge = (MockJudge() if args.model == "mock"
             else MLXJudge(args.model_id, args.temperature))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"judge_{stamp}.jsonl"

    print(f"attempts to judge : {len(attempts)}  x {args.k} repetitions "
          f"= {len(attempts)*args.k} judge calls")
    print(f"judge             : {judge.name} @ T={args.temperature}")
    print(f"writing           : {out_path}\n")

    with out_path.open("w") as f:
        f.write(json.dumps({"type": "judge_manifest", "judge": judge.name,
                            "temperature": args.temperature, "k": args.k,
                            "source_results": str(args.results_file),
                            "ts": time.time()}) + "\n")
        for i, a in enumerate(attempts, 1):
            q = questions[a["task_id"]]
            lang = "pt" if a["cell"].endswith("L1") else "en"
            question = q[lang] or q["en"]
            truth = bool(a.get("success"))
            verdicts = []
            for _ in range(args.k):
                verdicts.append(judge.verdict(question, q["expected"],
                                              a.get("got"), truth))
            f.write(json.dumps({
                "type": "judge_verdict",
                "task_id": a["task_id"], "cell": a["cell"],
                "seed_effective": a.get("seed_effective"),
                "attempt": a.get("attempt"),
                "objective_success": truth,
                "judge_verdicts": verdicts,
                "tok_total": a["tok_in_total"] + a["tok_out_total"],
            }) + "\n")
            if i % 25 == 0 or i == len(attempts):
                print(f"  [{i}/{len(attempts)}]")

    print(f"\ndone -> {out_path}")
    print("analyse with: python3 analysis/judge_analysis.py "
          f"{out_path}")


if __name__ == "__main__":
    main()
