"""Shared evaluation utilities — F1 score, exact match, eval harness."""

from __future__ import annotations

import re
import string


def normalize_answer(s: str) -> str:
    """Lowercase, strip articles, punctuation, and extra whitespace."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = sum(1 for t in pred_tokens if t in common) / len(pred_tokens)
    recall = sum(1 for t in gold_tokens if t in common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    """1.0 if normalized prediction equals normalized ground truth."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def run_eval(results: list[tuple[str, str]]) -> dict:
    """Evaluate a list of (prediction, ground_truth) pairs.

    Returns dict with avg_f1, avg_em, and per_question scores.
    """
    per_question: list[dict] = []
    for prediction, ground_truth in results:
        f1 = f1_score(prediction, ground_truth)
        em = exact_match(prediction, ground_truth)
        per_question.append({
            "prediction": prediction,
            "ground_truth": ground_truth,
            "f1": f1,
            "em": em,
        })
    n = len(per_question)
    avg_f1 = sum(q["f1"] for q in per_question) / n if n else 0.0
    avg_em = sum(q["em"] for q in per_question) / n if n else 0.0
    return {"avg_f1": avg_f1, "avg_em": avg_em, "per_question": per_question}
