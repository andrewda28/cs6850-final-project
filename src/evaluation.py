"""Binary evaluation with explicit undefined metrics and positive class 1."""

from collections.abc import Iterable
from statistics import mean

from src.schemas import Prediction, ReferenceLabel, Usage
from src.validation import index_unique, require_same_patients


def _divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _auroc(pairs: list[tuple[float, int]]) -> float | None:
    """Mann–Whitney rank statistic, assigning average ranks to tied scores."""
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(pairs)
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ordered[start:end])
        start = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate(labels: Iterable[ReferenceLabel], predictions: Iterable[Prediction]) -> dict:
    truth, predicted = index_unique(labels), index_unique(predictions)
    if not truth:
        raise ValueError("Cannot evaluate an empty cohort")
    require_same_patients(set(truth), set(predicted))
    if len({p.model_name for p in predicted.values()}) != 1:
        raise ValueError("Evaluate one model at a time")
    tn = fp = fn = tp = 0
    pairs = []
    for patient_id, reference in truth.items():
        prediction = predicted[patient_id]
        y, yhat = reference.label, prediction.predicted_label
        tn += int(y == 0 and yhat == 0)
        fp += int(y == 0 and yhat == 1)
        fn += int(y == 1 and yhat == 0)
        tp += int(y == 1 and yhat == 1)
        pairs.append((prediction.score, y))
    total_usage = sum((p.usage for p in predicted.values()), Usage())
    latencies = [p.latency_seconds for p in predicted.values()]
    review_times = [r.human_review_seconds for r in truth.values()]
    unknown_review = sum(t is None for t in review_times)
    routes: dict[str, int] = {}
    for prediction in predicted.values():
        routes[prediction.route] = routes.get(prediction.route, 0) + 1
    return {
        "n_patients": len(truth),
        "metrics": {
            "auroc": _auroc(pairs), "sensitivity": _divide(tp, tp + fn),
            "ppv": _divide(tp, tp + fp), "specificity": _divide(tn, tn + fp),
            "accuracy": (tp + tn) / len(truth), "f1": _divide(2 * tp, 2 * tp + fp + fn),
        },
        "confusion_matrix": {"labels": [0, 1], "rows": "true", "columns": "predicted",
                             "matrix": [[tn, fp], [fn, tp]], "tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "costs": {
            "latency_seconds_total": sum(latencies), "latency_seconds_mean": mean(latencies),
            "input_tokens": total_usage.input_tokens, "output_tokens": total_usage.output_tokens,
            "llm_calls": total_usage.llm_calls,
            "prediction_human_review_seconds": total_usage.human_review_seconds,
            "predictions_with_unknown_input_tokens": sum(p.usage.input_tokens is None for p in predicted.values()),
            "predictions_with_unknown_output_tokens": sum(p.usage.output_tokens is None for p in predicted.values()),
            "predictions_with_unknown_review_effort": sum(p.usage.human_review_seconds is None for p in predicted.values()),
        },
        # Shared cohort annotation cost, never add this once per model.
        "reference_review": {
            "seconds_total": None if unknown_review else sum(t for t in review_times if t is not None),
            "patients_with_unknown_effort": unknown_review,
        },
        "routes": routes,
    }
