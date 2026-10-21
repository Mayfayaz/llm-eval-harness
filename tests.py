"""Tests for llm-eval-harness. Run with: python tests.py (stdlib unittest)."""
import unittest

from eval_harness import run_eval
from eval_harness.metrics import exact_match, keyword_coverage, normalize, token_f1
from eval_harness.judge import HeuristicJudge
from eval_harness.models import MockModel, dispatch_model_fn


class TestMetrics(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("  The  QUICK, brown fox! "), "quick brown fox")

    def test_exact_match(self):
        self.assertEqual(exact_match("Paris", "paris."), 1.0)
        self.assertEqual(exact_match("Paris", "London"), 0.0)

    def test_token_f1(self):
        self.assertAlmostEqual(token_f1("the cat sat", "cat sat"), 1.0)
        self.assertEqual(token_f1("dog", "cat"), 0.0)
        # partial overlap -> between 0 and 1
        f1 = token_f1("the cat sat on the mat", "cat sat")
        self.assertTrue(0.0 < f1 < 1.0)
        self.assertEqual(token_f1("", ""), 1.0)
        self.assertEqual(token_f1("", "cat"), 0.0)

    def test_keyword_coverage(self):
        self.assertEqual(keyword_coverage("TCP is reliable", ["reliable", "udp"]), 0.5)
        self.assertEqual(keyword_coverage("anything", []), 1.0)


class TestJudge(unittest.TestCase):
    def test_heuristic_judge_perfect(self):
        j = HeuristicJudge()
        out = j.judge("q", "Paris is the capital of France", "Paris",
                      keywords=["paris"])
        self.assertEqual(out["relevance"], 5)
        self.assertEqual(out["completeness"], 5)
        self.assertIn("reasons", out)

    def test_heuristic_judge_bad(self):
        j = HeuristicJudge()
        out = j.judge("q", "I like turtles", "Paris", keywords=["paris"])
        self.assertLessEqual(out["relevance"], 2)
        self.assertEqual(out["completeness"], 1)


class TestRunner(unittest.TestCase):
    def test_run_eval_aggregates(self):
        tasks = [
            {"id": "t1", "prompt": "p1", "reference": "Paris", "keywords": ["paris"]},
            {"id": "t2", "prompt": "p2", "reference": "London", "keywords": ["london"]},
        ]
        model_fn = dispatch_model_fn(tasks, {"t1": "Paris", "t2": "Berlin"})
        report = run_eval(tasks, model_fn)
        s = report["summary"]
        self.assertEqual(s["n_tasks"], 2)
        self.assertEqual(s["mean_exact_match"], 0.5)
        self.assertEqual(report["results"][0]["exact_match"], 1.0)
        self.assertEqual(report["results"][1]["exact_match"], 0.0)
        self.assertTrue(all(r["latency_s"] >= 0 for r in report["results"]))

    def test_mock_model_default(self):
        m = MockModel({}).for_task("missing")
        self.assertEqual(m("anything"), "I don't know.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
