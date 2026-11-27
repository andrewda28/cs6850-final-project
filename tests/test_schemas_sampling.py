import random

import pytest

from src.sampling import sample_test_cohort
from src.schemas import Patient, Prediction, ReferenceLabel, Usage
from src.synthetic import synthetic_dataset
from src.validation import validate_no_leakage


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_scores(score):
    with pytest.raises(ValueError):
        Prediction("synthetic-p", "test", score, 1, 0)


@pytest.mark.parametrize("kwargs", [{"input_tokens": -1}, {"llm_calls": 1.5},
                                   {"output_tokens": True}, {"human_review_seconds": -2}])
def test_invalid_usage(kwargs):
    with pytest.raises(ValueError):
        Usage(**kwargs)


def test_unknown_usage_is_not_zero():
    assert Usage(None, 2, 1, None) + Usage(3, 4, 1, 5) == Usage(None, 6, 2, None)


def test_schema_validation():
    with pytest.raises(ValueError):
        Patient("")
    with pytest.raises(ValueError):
        ReferenceLabel("synthetic-p", 2)
    with pytest.raises(ValueError):
        Patient("synthetic-p", diagnoses=["invented"])


def test_sampling_reproducible_excluded_and_independent_of_order_and_global_rng():
    patients, _ = synthetic_dataset(50)
    development = {p.patient_id for p in patients[:10]}
    state = random.getstate()
    first = sample_test_cohort(patients, development, size=20, seed=42)
    assert random.getstate() == state
    assert first == sample_test_cohort(reversed(patients), development, size=20, seed=42)
    assert first != sample_test_cohort(patients, development, size=20, seed=43)
    assert len({p.patient_id for p in first}) == 20
    assert not development & {p.patient_id for p in first}


def test_duplicates_are_rejected_even_when_excluded():
    patient = Patient("synthetic-p")
    with pytest.raises(ValueError, match="Duplicate"):
        sample_test_cohort([patient, patient], {patient.patient_id}, size=1, seed=0)


@pytest.mark.parametrize("size", [0, -1, 25, True, 1.5])
def test_bad_sample_sizes(size):
    patients, _ = synthetic_dataset()
    with pytest.raises(ValueError):
        sample_test_cohort(patients, [], size=size, seed=0)


def test_direct_leakage_guard():
    with pytest.raises(ValueError, match="leakage"):
        validate_no_leakage(["synthetic-dev"], ["synthetic-dev"])
