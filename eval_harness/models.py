"""Model adapters: anything callable that maps prompt -> response.

MockModel lets you demo and test the harness with no API key. It can
simulate failure modes (verbosity, hallucination, refusal) so you can
see how the metrics respond to each one.
"""
from __future__ import annotations


class MockModel:
    """Deterministic scripted model: answers are looked up by task id."""

    def __init__(self, answers: dict[str, str], default: str = "I don't know."):
        self.answers = answers
        self.default = default
        self.task_id: str | None = None

    def for_task(self, task_id: str) -> "MockModel":
        """Return a copy bound to one task id."""
        m = MockModel(self.answers, self.default)
        m.task_id = task_id
        return m

    def __call__(self, prompt: str) -> str:
        return self.answers.get(self.task_id or "", self.default)


def dispatch_model_fn(tasks: list[dict], answers: dict[str, str]):
    """Build a single prompt->response fn for run_eval over an ordered task list."""
    models = [MockModel(answers).for_task(t["id"]) for t in tasks]
    state = {"i": 0}

    def fn(prompt: str) -> str:
        m = models[state["i"]]
        state["i"] += 1
        return m(prompt)

    return fn
