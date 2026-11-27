"""Evaluate all classifiers on one sampled, fully labeled, held-out cohort."""

from collections.abc import Iterable
from dataclasses import dataclass
import platform

from src.evaluation import evaluate
from src.models.base import Classifier
from src.sampling import sample_test_cohort
from src.schemas import Patient, Prediction, ReferenceLabel, identifier
from src.validation import index_unique, require_same_patients, validate_no_leakage


@dataclass(frozen=True)
class ExperimentResult:
    # These patient-level values are restricted artifacts; keep them out of Git/logs.
    cohort_ids: tuple[str, ...]
    predictions: dict[str, tuple[Prediction, ...]]
    summary: dict


def run_experiment(
    patients: Iterable[Patient], labels: Iterable[ReferenceLabel], development_ids: Iterable[str],
    models: Iterable[Classifier], *, size: int, seed: int, data_version: str,
) -> ExperimentResult:
    identifier(data_version)
    development = set(development_ids)
    cohort = sample_test_cohort(patients, development, size=size, seed=seed)
    cohort_ids = tuple(p.patient_id for p in cohort)
    validate_no_leakage(cohort_ids, development)
    references = index_unique(labels)
    if not set(cohort_ids) <= set(references):
        raise ValueError("Missing human-reviewed labels for sampled cohort")
    selected_labels = [references[patient_id] for patient_id in cohort_ids]
    classifiers = list(models)
    names = [m.name for m in classifiers]
    for name in names:
        identifier(name)
    if not classifiers or len(set(names)) != len(names):
        raise ValueError("Provide classifiers with unique names")
    predictions, reports = {}, {}
    configurations = {m.name: m.describe() for m in classifiers}
    for model in classifiers:
        outputs = tuple(model.predict(patient) for patient in cohort)
        indexed = index_unique(outputs)
        require_same_patients(set(cohort_ids), set(indexed))
        if any(p.model_name != model.name for p in outputs):
            raise ValueError("Prediction model name does not match classifier")
        predictions[model.name] = outputs
        reports[model.name] = evaluate(selected_labels, outputs)
    return ExperimentResult(cohort_ids, predictions, {
        "schema_version": 1, "data_version": data_version, "seed": seed, "sample_size": size,
        "python_version": platform.python_version(),
        "sampling": "uniform without replacement over sorted eligible patient IDs",
        "undefined_metrics": "null means denominator zero or AUROC lacks both classes",
        "models": configurations, "results": reports,
    })
