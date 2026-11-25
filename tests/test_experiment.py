import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.experiment import run_experiment
from src.llm_client import MockClient
from src.models import AdaptiveAgent, ICDBaseline, MultiAgent, SingleLLM
from src.schemas import Prediction
from src.synthetic import SYNTHETIC_POSITIVE, synthetic_dataset


def models():
    client = MockClient()
    return [ICDBaseline(frozenset({SYNTHETIC_POSITIVE}), code_set_version="synthetic-v1"),
            SingleLLM(client), MultiAgent(client), AdaptiveAgent(client)]


def test_every_model_uses_identical_cohort_and_summary_has_no_patient_records():
    patients, labels = synthetic_dataset()
    development = {p.patient_id for p in patients[:4]}
    result = run_experiment(patients, labels, development, models(), size=12, seed=6850, data_version="synthetic-v1")
    assert not development & set(result.cohort_ids)
    assert len(result.predictions) == 4
    for predictions in result.predictions.values():
        assert tuple(p.patient_id for p in predictions) == result.cohort_ids
    summary = json.dumps(result.summary, allow_nan=False)
    assert not any(p.patient_id in summary or p.clinical_text in summary for p in patients)
    again = run_experiment(reversed(patients), labels, development, models(), size=12, seed=6850, data_version="synthetic-v1")
    assert again.cohort_ids == result.cohort_ids
    for name in result.predictions:
        assert [p.score for p in result.predictions[name]] == [p.score for p in again.predictions[name]]


def test_runner_rejects_missing_labels_and_duplicate_models_before_inference():
    patients, labels = synthetic_dataset()
    with pytest.raises(ValueError, match="Missing"):
        run_experiment(patients, [], [], models(), size=2, seed=0, data_version="synthetic")
    with pytest.raises(ValueError, match="unique"):
        run_experiment(patients, labels, [], [models()[0], models()[0]], size=2, seed=0, data_version="synthetic")


@pytest.mark.parametrize("wrong_id", [True, False])
def test_runner_rejects_misaligned_model_output(wrong_id):
    class BrokenModel:
        name = "broken"

        def describe(self):
            return {}

        def predict(self, patient):
            return Prediction("synthetic-unexpected" if wrong_id else patient.patient_id,
                              self.name if wrong_id else "wrong-model", 0, 0, 0)

    patients, labels = synthetic_dataset()
    with pytest.raises(ValueError):
        run_experiment(patients, labels, [], [BrokenModel()], size=1, seed=0, data_version="synthetic")


def test_cli_synthetic_smoke():
    completed = subprocess.run([sys.executable, "run_experiment.py", "--size", "8"],
                               check=True, capture_output=True, text=True)
    summary = json.loads(completed.stdout)
    assert summary["data_version"] == "fully-synthetic-v1"
    assert len(summary["results"]) == 4
    assert all(r["n_patients"] == 8 for r in summary["results"].values())


def test_sensitive_paths_are_ignored():
    root = Path(__file__).resolve().parents[1]
    paths = ["data/patients.txt", "outputs/predictions.txt", "notes.ipynb", "nested/patients.csv",
             "nested/predictions.json", ".env.local", "credentials.txt", "private/review.txt"]
    completed = subprocess.run(["git", "check-ignore", "--stdin"], input="\n".join(paths),
                               text=True, capture_output=True, check=True, cwd=root)
    assert set(completed.stdout.splitlines()) == set(paths)
