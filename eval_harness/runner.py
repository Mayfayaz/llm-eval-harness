"""Eval runner: executes a model over a task dataset and scores every output."""
from __future__ import annotations

import json
import time
from typing import Callable

from . import metrics
from .judge import HeuristicJudge

ModelFn = Callable[[str], str]


def run_eval(tasks: list[dict], model_fn: ModelFn, judge=None) -> dict:
    """Run `model_fn(prompt)` for each task and score the output.

    Each task: {"id", "prompt", "reference", "keywords" (optional)}.
    Returns a dict with per-task results and aggregate summary.
    """
    judge = judge or HeuristicJudge()
    results = []
    for task in tasks:
        start = time.perf_counter()
        response = model_fn(task["prompt"])
        latency_s = time.perf_counter() - start
        keywords = task.get("keywords", [])
        task_result = {
            "id": task["id"],
            "prompt": task["prompt"],
            "reference": task["reference"],
            "response": response,
            "latency_s": round(latency_s, 3),
            "exact_match": metrics.exact_match(response, task["reference"]),
            "token_f1": round(metrics.token_f1(response, task["reference"]), 3),
            "keyword_coverage": round(metrics.keyword_coverage(response, keywords), 3),
            "length_chars": metrics.answer_length_chars(response),
            "judge": judge.judge(task["prompt"], response, task["reference"], keywords),
        }
        results.append(task_result)

    n = len(results)
    summary = {
        "n_tasks": n,
        "mean_exact_match": round(sum(r["exact_match"] for r in results) / n, 3),
        "mean_token_f1": round(sum(r["token_f1"] for r in results) / n, 3),
        "mean_keyword_coverage": round(
            sum(r["keyword_coverage"] for r in results) / n, 3
        ),
        "mean_relevance": round(
            sum(r["judge"]["relevance"] for r in results) / n, 2
        ),
        "mean_completeness": round(
            sum(r["judge"]["completeness"] for r in results) / n, 2
        ),
        "mean_latency_s": round(sum(r["latency_s"] for r in results) / n, 3),
        "mean_length_chars": round(sum(r["length_chars"] for r in results) / n, 1),
    }
    return {"summary": summary, "results": results}


def render_markdown_report(report: dict) -> str:
    """Render the eval report as a Markdown table for the README/results."""
    s = report["summary"]
    lines = [
        "# Eval Report",
        "",
        f"Tasks: **{s['n_tasks']}** · mean EM: **{s['mean_exact_match']}** · "
        f"mean token F1: **{s['mean_token_f1']}** · "
        f"mean judge relevance: **{s['mean_relevance']}/5** · "
        f"mean judge completeness: **{s['mean_completeness']}/5**",
        "",
        "| id | EM | token F1 | kw coverage | relevance | completeness | latency (s) |",
        "|----|----|----------|-------------|-----------|--------------|-------------|",
    ]
    for r in report["results"]:
        lines.append(
            f"| {r['id']} | {r['exact_match']:.0f} | {r['token_f1']} | "
            f"{r['keyword_coverage']} | {r['judge']['relevance']} | "
            f"{r['judge']['completeness']} | {r['latency_s']} |"
        )
    return "\n".join(lines) + "\n"


def save_report(report: dict, path: str) -> None:
    """Write the full report (results + summary) as JSON."""
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
