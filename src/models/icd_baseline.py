"""Exact-match baseline. The team must supply a reviewed, versioned ICD code set."""

from time import perf_counter

from src.models.base import ScoredClassifier
from src.schemas import DiagnosisCode, Patient, Prediction, Usage, identifier


class ICDBaseline(ScoredClassifier):
    def __init__(self, positive_codes: frozenset[DiagnosisCode], *, code_set_version: str):
        super().__init__("icd_baseline")
        identifier(code_set_version)
        if not positive_codes or not all(isinstance(c, DiagnosisCode) for c in positive_codes):
            raise ValueError("Provide a nonempty reviewed code set (or explicit synthetic fixture codes)")
        self.positive_codes = frozenset(positive_codes)
        self.code_set_version = code_set_version

    def predict(self, patient: Patient) -> Prediction:
        started = perf_counter()
        score = float(bool(set(patient.diagnoses) & self.positive_codes))
        return self._prediction(patient, score, Usage(), started, "diagnosis_codes")

    def describe(self) -> dict:
        return {"name": self.name, "code_set_version": self.code_set_version, "threshold": self.threshold,
                "positive_codes": [{"system": c.system, "code": c.code}
                                   for c in sorted(self.positive_codes, key=lambda c: (c.system, c.code))]}
