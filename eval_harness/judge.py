"""Pluggable judges for rubric-based scoring of model outputs.

The default judge is fully deterministic and needs no API key, so the
harness (and its tests) run everywhere. An LLM-backed judge adapter is
included as a documented optional path.
"""
from __future__ import annotations

import json
import os
import urllib.request

from .metrics import keyword_coverage, normalize


class HeuristicJudge:
    """Deterministic rubric judge: scores 1-5 on relevance and completeness.

    Not a substitute for a real LLM-as-judge — it scores against the
    reference answer and required keywords — but it gives you a working,
    reproducible judge pipeline to build on.
    """

    def judge(self, prompt: str, response: str, reference: str,
              keywords: list[str] | None = None) -> dict:
        keywords = keywords or []
        coverage = keyword_coverage(response, keywords)
        # Relevance: how much of the reference's content shows up in the answer.
        ref_tokens = set(normalize(reference).split())
        resp_tokens = set(normalize(response).split())
        overlap = len(ref_tokens & resp_tokens) / max(len(ref_tokens), 1)
        relevance = 1 + round(4 * min(1.0, overlap))
        # Completeness: all required keywords covered?
        completeness = 1 + round(4 * coverage)
        reasons = (
            f"Reference-token overlap {overlap:.0%}; "
            f"keyword coverage {coverage:.0%} ({keywords or 'none required'})."
        )
        return {
            "relevance": relevance,
            "completeness": completeness,
            "reasons": reasons,
        }


class OpenAICompatibleJudge:
    """LLM-as-judge via any OpenAI-compatible /chat/completions endpoint.

    Requires: OAI_BASE_URL (optional, defaults to api.openai.com) and
    OAI_API_KEY environment variables. Only stdlib is used.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.api_key = os.environ.get("OAI_API_KEY", "")
        self.base_url = os.environ.get(
            "OAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model

    def judge(self, prompt: str, response: str, reference: str,
              keywords: list[str] | None = None) -> dict:
        if not self.api_key:
            raise RuntimeError("OAI_API_KEY is not set; cannot use LLM judge.")
        rubric = (
            "You are an impartial evaluator. Score the model ANSWER on two "
            "criteria, each an integer 1-5: relevance (does it address the "
            "QUESTION?) and completeness (does it cover the REFERENCE answer?). "
            "Reply with ONLY a JSON object: "
            '{"relevance": N, "completeness": N, "reasons": "..."}'
        )
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": rubric},
                {"role": "user", "content":
                 f"QUESTION: {prompt}\nREFERENCE: {reference}\nANSWER: {response}"},
            ],
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = json.loads(resp.read())["choices"][0]["message"]["content"]
        return json.loads(content)
