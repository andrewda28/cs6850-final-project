import gzip

import pandas as pd
import pytest

from collect_review_labels import collect_review_labels
from build_development_exclusions import build_development_exclusions
from prepare_test_cohort import prepare_test_cohort
from run_real_experiment import load_test_patients

from src.models import ICDBaseline
from src.schemas import DiagnosisCode
from src.evaluation import evaluate


def test_development_exclusions(tmp_path):

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    pd.DataFrame({
        "subject_id": ["1", "2", "3"]
    }).to_csv(first, index=False)

    pd.DataFrame({
        "subject_id": ["3", "4"]
    }).to_csv(second, index=False)

    result = build_development_exclusions(
        [first, second]
    )

    assert result == {"1", "2", "3", "4"}


def test_prepare_test_cohort(tmp_path):

    positive = tmp_path / "positive.csv"
    negative = tmp_path / "negative.csv"
    exclusions = tmp_path / "exclusions.csv"

    def make_data(ids):
        return pd.DataFrame({
            "subject_id": ids,
            "hadm_id": [f"H{i}" for i in ids],
            "text": ["Synthetic clinical note"] * len(ids),
            "all_icd_codes": ["G300"] * len(ids)
        })

    make_data(["1", "2", "3", "4"]).to_csv(
        positive, index=False
    )

    make_data(["5", "6", "7", "8"]).to_csv(
        negative, index=False
    )

    pd.DataFrame({
        "subject_id": ["1", "5"]
    }).to_csv(exclusions, index=False)

    cohort = prepare_test_cohort(
        positive,
        negative,
        exclusions,
        size_per_group=2,
        seed=6850
    )

    assert len(cohort) == 4
    assert cohort["subject_id"].is_unique
    assert not set(cohort["subject_id"]) & {"1", "5"}

    assert (
        cohort["sampling_group"] == "positive_enriched"
    ).sum() == 2

    assert (
        cohort["sampling_group"] == "negative_enriched"
    ).sum() == 2

    repeated = prepare_test_cohort(
        positive,
        negative,
        exclusions,
        size_per_group=2,
        seed=6850
    )

    pd.testing.assert_frame_equal(cohort, repeated)


@pytest.fixture
def completed_reviews(tmp_path):

    mapping = pd.DataFrame({
        "review_id": [f"TEST-{i:03d}" for i in range(50)],
        "subject_id": [str(i) for i in range(50)],
        "hadm_id": [f"H{i}" for i in range(50)],
        "reviewer": ["reviewer"] * 50
    })

    mapping.to_csv(
        tmp_path / "assignment_mapping.csv",
        index=False
    )

    reviews = pd.DataFrame({
        "review_id": mapping["review_id"],
        "human_label": [1] * 20 + [0] * 25 + [-1] * 5,
        "reasoning": ["Synthetic review"] * 50,
        "review_seconds": [60] * 50
    })

    reviews.to_csv(
        tmp_path / "reviewer_review.csv",
        index=False
    )

    return tmp_path


def test_collect_review_labels(completed_reviews):

    labels = collect_review_labels(completed_reviews)

    assert len(labels) == 50
    assert (labels["human_label"] == 1).sum() == 20
    assert (labels["human_label"] == 0).sum() == 25
    assert (labels["human_label"] == -1).sum() == 5


def test_reject_incomplete_labels(completed_reviews):

    path = completed_reviews / "reviewer_review.csv"

    reviews = pd.read_csv(path, dtype=str)

    reviews.loc[0, "human_label"] = ""

    reviews.to_csv(path, index=False)

    with pytest.raises(ValueError):
        collect_review_labels(completed_reviews)


def test_reject_duplicate_reviews(completed_reviews):

    path = completed_reviews / "reviewer_review.csv"

    reviews = pd.read_csv(path, dtype=str)

    reviews.loc[1, "review_id"] = reviews.loc[0, "review_id"]

    reviews.to_csv(path, index=False)

    with pytest.raises(ValueError):
        collect_review_labels(completed_reviews)


def test_end_to_end_evaluation(tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)

    base = tmp_path / "data" / "processed"
    base.mkdir(parents=True)

    cohort = pd.DataFrame({
        "subject_id": [str(i) for i in range(50)],
        "hadm_id": [str(100 + i) for i in range(50)],
        "text": ["Synthetic clinical note"] * 50,
        "all_icd_codes": ["G300"] * 20 + ["4019"] * 30
    })

    cohort.to_csv(
        base / "test_cohort.csv",
        index=False
    )

    labels = cohort[["subject_id", "hadm_id"]].copy()

    labels["human_label"] = [1] * 20 + [0] * 25 + [-1] * 5
    labels["review_seconds"] = 60
    labels["reviewer"] = "synthetic_reviewer"

    labels.to_csv(
        base / "test_reference_labels.csv",
        index=False
    )

    pd.DataFrame({
        "subject_id": ["999"]
    }).to_csv(
        base / "development_exclusions.csv",
        index=False
    )

    diagnoses = cohort[
        ["subject_id", "hadm_id"]
    ].copy()

    diagnoses["seq_num"] = 1
    diagnoses["icd_code"] = cohort["all_icd_codes"]
    diagnoses["icd_version"] = [10] * 20 + [9] * 30

    diagnosis_file = base / "diagnoses.csv.gz"

    with gzip.open(diagnosis_file, "wt") as f:
        diagnoses.to_csv(f, index=False)

    patients, references = load_test_patients(
        diagnosis_file
    )

    assert len(patients) == 45
    assert len(references) == 45

    model = ICDBaseline(
        frozenset({DiagnosisCode("ICD-10", "G300")}),
        code_set_version="synthetic-test-v1"
    )

    predictions = [
        model.predict(patient)
        for patient in patients
    ]

    results = evaluate(references, predictions)

    assert results["metrics"]["accuracy"] == 1.0
    assert results["metrics"]["f1"] == 1.0
    assert results["metrics"]["sensitivity"] == 1.0
    assert results["metrics"]["specificity"] == 1.0