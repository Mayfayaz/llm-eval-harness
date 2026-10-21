"""Demo: evaluate a mock model on the bundled 8-task dataset.

Runs entirely offline — no API keys needed. Simulates a decent-but-
imperfect model (one hallucinated answer, one refusal, one verbose answer)
so the report shows how each metric reacts.

Usage:
    python demo.py                      # heuristic judge (default)
    python demo.py --report out.md      # also write a Markdown report
    python demo.py --llm-judge          # use an OpenAI-compatible LLM judge
                                        # (needs OAI_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import sys

from eval_harness import run_eval, render_markdown_report, save_report
from eval_harness.judge import OpenAICompatibleJudge
from eval_harness.models import dispatch_model_fn

# A "decent but imperfect" model: mostly right, with three failure modes.
ANSWERS = {
    "capital-france": "Paris",
    "photosynthesis": "Photosynthesis produces glucose and oxygen.",
    "python-list": "Call my_list.append(item) to add an item to the end of a Python list.",
    "http-404": "The capital of France is Paris.",  # hallucination (wrong task)
    "water-boiling": "I don't know.",  # refusal
    "git-commit": "Use git commit -m \"your message\" to record staged changes.",
    "binary-search": (
        "Binary search has O(log n) time complexity because each step halves "
        "the search space. This was first described in the context of sorted "
        "arrays, where it dramatically outperforms linear search at O(n), and "
        "it remains a fundamental algorithm taught in every CS curriculum."
    ),  # verbose but correct
    "tcp-vs-udp": "TCP is reliable and connection-oriented; UDP is connectionless and faster.",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=None, help="Write Markdown report to this path")
    parser.add_argument("--json", default="results.json", help="Write full results JSON here")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Use OpenAI-compatible LLM judge (needs OAI_API_KEY)")
    args = parser.parse_args()

    with open("data/eval_dataset.json") as f:
        tasks = json.load(f)

    model_fn = dispatch_model_fn(tasks, ANSWERS)
    judge = OpenAICompatibleJudge() if args.llm_judge else None
    report = run_eval(tasks, model_fn, judge=judge)

    md = render_markdown_report(report)
    print(md)
    save_report(report, args.json)
    print(f"Full results written to {args.json}")
    if args.report:
        with open(args.report, "w") as f:
            f.write(md)
        print(f"Markdown report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
