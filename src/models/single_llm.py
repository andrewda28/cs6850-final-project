"""Diagnosis and text assessment with one LLM call; prompt is a research scaffold."""

from time import perf_counter

from src.llm_client import LLMClient
from src.models.base import ScoredClassifier
from src.schemas import Patient, Prediction


class SingleLLM(ScoredClassifier):
    def __init__(self, client: LLMClient, *, threshold: float = 0.5):
        super().__init__("single_llm", threshold)
        self.client = client

    def predict(self, patient: Patient) -> Prediction:
        started = perf_counter()
        response = self.client.score(patient, "integrated diagnosis and text review")
        return self._prediction(patient, response.score, response.usage, started, "single")

    def describe(self) -> dict:
        return {"name": self.name, "threshold": self.threshold, "client": self.client.describe()}
