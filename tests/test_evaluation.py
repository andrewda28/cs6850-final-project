import itertools

import pytest

from src.evaluation import evaluate
from src.schemas import Prediction, ReferenceLabel, Usage


def records(truth, scores):
    labels = [ReferenceLabel(f"synthetic-{i}", y, 2.0) for i, y in enumerate(truth)]
    predictions = [Prediction(f"synthetic-{i}", "test", s, int(s >= 0.5), 0.1, Usage(2, 3, 1, 1.0))
                   for i, s in enumerate(scores)]
    return labels, predictions


def test_hand_calculated_metrics_and_effort():
    labels, predictions = records([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    report = evaluate(labels, reversed(predictions))
    assert report["metrics"] == {"auroc": 0.75, "sensitivity": 0.5, "ppv": 0.5,
                                 "specificity": 0.5, "accuracy": 0.5, "f1": 0.5}
    assert report["confusion_matrix"]["matrix"] == [[1, 1], [1, 1]]
    assert report["costs"]["input_tokens"] == 8
    assert report["costs"]["output_tokens"] == 12
    assert report["costs"]["llm_calls"] == 4
    assert report["costs"]["prediction_human_review_seconds"] == 4
    assert report["reference_review"]["seconds_total"] == 8
    assert report["costs"]["latency_seconds_total"] == pytest.approx(0.4)


@pytest.mark.parametrize("scores", list(itertools.product([0.0, 0.5, 1.0], repeat=4)))
def test_auc_matches_pairwise_definition_including_ties(scores):
    labels, predictions = records([0, 1, 0, 1], scores)
    expected = sum((scores[p] > scores[n]) + 0.5 * (scores[p] == scores[n])
                   for p in [1, 3] for n in [0, 2]) / 4
    assert evaluate(labels, predictions)["metrics"]["auroc"] == expected


def test_undefined_metrics_are_explicit():
    labels, predictions = records([0, 0], [0.1, 0.2])
    metrics = evaluate(labels, predictions)["metrics"]
    assert metrics == {"auroc": None, "sensitivity": None, "ppv": None,
                       "specificity": 1.0, "accuracy": 1.0, "f1": None}
    labels, predictions = records([1, 1], [0.1, 0.2])
    assert evaluate(labels, predictions)["metrics"]["specificity"] is None
    assert evaluate(labels, predictions)["metrics"]["f1"] == 0


def test_unknown_costs_are_explicit():
    report = evaluate([ReferenceLabel("synthetic-p", 1)],
                      [Prediction("synthetic-p", "test", 0.7, 1, 0, Usage(None, None, 1, None))])
    assert report["costs"]["input_tokens"] is None
    assert report["costs"]["predictions_with_unknown_input_tokens"] == 1
    assert report["costs"]["prediction_human_review_seconds"] is None
    assert report["reference_review"] == {"seconds_total": None, "patients_with_unknown_effort": 1}


def test_missing_extra_duplicate_mixed_and_empty_predictions_rejected():
    labels, predictions = records([0, 1], [0.1, 0.9])
    for invalid in [predictions[:1], predictions + [predictions[0]],
                    predictions + [Prediction("synthetic-extra", "test", 0, 0, 0)],
                    [predictions[0], Prediction("synthetic-1", "other", 1, 1, 0)]]:
        with pytest.raises(ValueError):
            evaluate(labels, invalid)
    with pytest.raises(ValueError):
        evaluate(labels + [labels[0]], predictions)
    with pytest.raises(ValueError):
        evaluate([], [])
