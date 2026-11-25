"""Shared classifier interface and output construction."""

from time import perf_counter
from typing import Protocol

from src.schemas import Patient, Prediction, Usage, probability


class Classifier(Protocol):
    name: str

    def predict(self, patient: Patient) -> Prediction: ...

    def describe(self) -> dict: ...


class ScoredClassifier:
    def __init__(self, name: str, threshold: float = 0.5):
        probability(threshold)
        self.name = name
        self.threshold = threshold

    def _prediction(self, patient: Patient, score: float, usage: Usage, started: float, route: str) -> Prediction:
        return Prediction(patient.patient_id, self.name, score, int(score >= self.threshold),
                          perf_counter() - started, usage, route)
