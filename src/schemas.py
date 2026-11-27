"""Internal research contracts, not MIMIC table or column definitions."""

from dataclasses import dataclass, field
from math import isfinite


def nonnegative(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def probability(value: float) -> None:
    nonnegative(value, "score")
    if value > 1:
        raise ValueError("score must be in [0, 1]")


def identifier(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Identifiers must be nonempty strings")


def binary(value: int) -> None:
    if type(value) is not int or value not in (0, 1):
        raise ValueError("Labels must be integer 0 or 1")


@dataclass(frozen=True)
class DiagnosisCode:
    system: str
    code: str

    def __post_init__(self) -> None:
        identifier(self.system)
        identifier(self.code)


@dataclass(frozen=True)
class Patient:
    patient_id: str
    diagnoses: tuple[DiagnosisCode, ...] = ()
    clinical_text: str = ""

    def __post_init__(self) -> None:
        identifier(self.patient_id)
        if not isinstance(self.diagnoses, tuple) or not all(isinstance(c, DiagnosisCode) for c in self.diagnoses):
            raise ValueError("diagnoses must be a tuple of DiagnosisCode")
        if not isinstance(self.clinical_text, str):
            raise ValueError("clinical_text must be a string")


@dataclass(frozen=True)
class ReferenceLabel:
    patient_id: str
    label: int
    # None means not recorded; zero means explicitly no effort.
    human_review_seconds: float | None = None

    def __post_init__(self) -> None:
        identifier(self.patient_id)
        binary(self.label)
        if self.human_review_seconds is not None:
            nonnegative(self.human_review_seconds, "human_review_seconds")


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    llm_calls: int = 0
    human_review_seconds: float | None = 0.0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "llm_calls"):
            value = getattr(self, name)
            if value is None and name != "llm_calls":
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.human_review_seconds is not None:
            nonnegative(self.human_review_seconds, "human_review_seconds")

    def __add__(self, other: "Usage") -> "Usage":
        def add(a: int | float | None, b: int | float | None):
            return None if a is None or b is None else a + b
        return Usage(
            add(self.input_tokens, other.input_tokens),
            add(self.output_tokens, other.output_tokens),
            self.llm_calls + other.llm_calls,
            add(self.human_review_seconds, other.human_review_seconds),
        )


@dataclass(frozen=True)
class Prediction:
    patient_id: str
    model_name: str
    score: float
    predicted_label: int
    latency_seconds: float
    usage: Usage = field(default_factory=Usage)
    route: str = "direct"

    def __post_init__(self) -> None:
        identifier(self.patient_id)
        identifier(self.model_name)
        identifier(self.route)
        probability(self.score)
        binary(self.predicted_label)
        nonnegative(self.latency_seconds, "latency_seconds")
        if not isinstance(self.usage, Usage):
            raise ValueError("usage must be Usage")
