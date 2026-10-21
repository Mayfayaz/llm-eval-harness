"""llm-eval-harness: a tiny, dependency-free eval harness for LLM outputs."""
from .runner import run_eval, render_markdown_report, save_report

__all__ = ["run_eval", "render_markdown_report", "save_report"]
__version__ = "0.1.0"
