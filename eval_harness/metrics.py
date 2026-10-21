"""Core evaluation metrics for LLM outputs.

All metrics are deterministic and dependency-free so the harness runs
anywhere with just the Python standard library.
"""
import re
import string


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles, collapse whitespace (SQuAD-style)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if normalized strings are identical, else 0.0."""
    return 1.0 if normalize(prediction) == normalize(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 over normalized tokens (SQuAD-style)."""
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = 0
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    for t in pred_tokens:
        if ref_counts.get(t, 0) > 0:
            ref_counts[t] -= 1
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(prediction: str, keywords: list[str]) -> float:
    """Fraction of required keywords present in the prediction."""
    if not keywords:
        return 1.0
    pred = normalize(prediction)
    hits = sum(1 for kw in keywords if normalize(kw) in pred)
    return hits / len(keywords)


def answer_length_chars(prediction: str) -> int:
    """Raw character length of the model answer (cost/verbosity signal)."""
    return len(prediction)
