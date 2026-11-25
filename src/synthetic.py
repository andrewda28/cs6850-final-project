"""Entirely invented fixtures; SYNTH codes have no clinical meaning."""

from src.schemas import DiagnosisCode, Patient, ReferenceLabel


SYNTHETIC_POSITIVE = DiagnosisCode("SYNTH", "DEMO_POSITIVE")


def synthetic_dataset(count: int = 24) -> tuple[list[Patient], list[ReferenceLabel]]:
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    patients, labels = [], []
    for i in range(count):
        patient_id = f"synthetic-{i:04d}"
        patients.append(Patient(
            patient_id,
            (SYNTHETIC_POSITIVE,) if i % 3 == 0 else (),
            f"Entirely synthetic fixture {i}; no actual patient or clinical source.",
        ))
        labels.append(ReferenceLabel(patient_id, i % 2, 0.0))
    return patients, labels
