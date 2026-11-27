"""Minimal role-based ensemble, not a validated clinical multi-agent protocol."""

from time import perf_counter

from src.llm_client import LLMClient
from src.models.base import ScoredClassifier
from src.schemas import Patient, Prediction, Usage


class MultiAgent(ScoredClassifier):
    roles = ("diagnosis evidence review", "text evidence and alternative explanations review")

    def __init__(self, client: LLMClient, *, threshold: float = 0.5):
        super().__init__("multi_agent", threshold)
        self.client = client

    def predict(self, patient: Patient) -> Prediction:
        started = perf_counter()
        responses = [self.client.score(patient, role) for role in self.roles]
        score = sum(r.score for r in responses) / len(responses)
        usage = sum((r.usage for r in responses), Usage())
        return self._prediction(patient, score, usage, started, "multi")

    def describe(self) -> dict:
        return {"name": self.name, "threshold": self.threshold, "roles": list(self.roles),
                "aggregation": "arithmetic_mean", "client": self.client.describe()}
