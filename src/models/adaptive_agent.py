"""Escalate inclusive score-band ambiguity from one call to the role ensemble."""

from time import perf_counter

from src.llm_client import LLMClient
from src.models.base import ScoredClassifier
from src.models.multi_agent import MultiAgent
from src.models.single_llm import SingleLLM
from src.schemas import Patient, Prediction, probability


class AdaptiveAgent(ScoredClassifier):
    def __init__(self, client: LLMClient, *, expensive_client: LLMClient | None = None,
                 lower: float = 0.35, upper: float = 0.65, threshold: float = 0.5):
        super().__init__("adaptive_agent", threshold)
        probability(lower)
        probability(upper)
        if not lower <= threshold <= upper or lower == upper:
            raise ValueError("Ambiguity band must have positive width and contain threshold")
        self.lower, self.upper = lower, upper
        self.single = SingleLLM(client, threshold=threshold)
        self.multi = MultiAgent(expensive_client if expensive_client is not None else client, threshold=threshold)

    def predict(self, patient: Patient) -> Prediction:
        started = perf_counter()
        initial = self.single.predict(patient)
        if self.lower <= initial.score <= self.upper:
            final = self.multi.predict(patient)
            return self._prediction(patient, final.score, initial.usage + final.usage, started, "single_to_multi")
        return self._prediction(patient, initial.score, initial.usage, started, "single_only")

    def describe(self) -> dict:
        return {"name": self.name, "threshold": self.threshold, "lower": self.lower, "upper": self.upper,
                "escalation": "inclusive_band; ensemble replaces initial score",
                "single": self.single.describe(), "multi": self.multi.describe()}
