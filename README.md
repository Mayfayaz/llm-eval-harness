# llm-eval-harness

A **dependency-free eval harness** for LLM outputs: score a model over a
task dataset with classic string metrics *and* a pluggable judge
(LLM-as-judge), then get a Markdown report plus full JSON results.

Built as a daily AI-engineering exercise in what a minimal eval loop looks
like — the same shape as tools like [EleutherAI's lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness),
stripped down to ~200 lines you can read in one sitting.

## What it does

For each task (`prompt` → `reference` answer + required keywords) it runs a
model function and computes:

| Metric | What it tells you |
|---|---|
| `exact_match` | Strict correctness (SQuAD-style normalization) |
| `token_f1` | Partial credit for near-misses |
| `keyword_coverage` | Did the answer hit the required facts? |
| `judge relevance / completeness` | 1–5 rubric scores from a judge |
| `latency_s`, `length_chars` | Cost and verbosity signals |

## Quickstart

```bash
# runs fully offline — no API keys needed
python3 demo.py

# also write a Markdown report
python3 demo.py --report report.md

# run the tests
python3 tests.py
```

The demo evaluates a scripted `MockModel` that is deliberately imperfect —
one hallucinated answer, one refusal, one verbose-but-correct answer — so you
can see how each metric reacts.

## Bring your own model

`run_eval(tasks, model_fn)` takes any `prompt -> str` callable:

```python
from eval_harness import run_eval
import json

tasks = json.load(open("data/eval_dataset.json"))

def my_model(prompt: str) -> str:
    ...  # call your API / local model here

report = run_eval(tasks, my_model)
```

## Bring your own judge

The default `HeuristicJudge` is deterministic and offline (it scores against
the reference answer and keywords — great for CI, not a real critic). For a
genuine LLM-as-judge, pass `--llm-judge` to the demo (or instantiate
`OpenAICompatibleJudge`) with `OAI_API_KEY` set; it speaks any
OpenAI-compatible `/chat/completions` endpoint via `OAI_BASE_URL`.

## Layout

```
eval_harness/
  metrics.py   # exact_match, token F1, keyword coverage
  judge.py     # HeuristicJudge + OpenAICompatibleJudge (LLM-as-judge)
  runner.py    # run_eval(), Markdown report, JSON export
  models.py    # MockModel for offline demos/tests
data/eval_dataset.json   # 8-task sample dataset
demo.py        # runnable demo (offline by default)
tests.py       # stdlib unittest suite
```

## License

MIT — see [LICENSE](LICENSE).
