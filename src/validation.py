"""Fail closed on cohort and alignment errors without logging identifiers."""

from collections.abc import Iterable
from typing import Protocol, TypeVar

from src.schemas import identifier


class Identified(Protocol):
    patient_id: str


T = TypeVar("T", bound=Identified)


def index_unique(records: Iterable[T]) -> dict[str, T]:
    result = {}
    for record in records:
        if record.patient_id in result:
            raise ValueError("Duplicate patient detected")
        result[record.patient_id] = record
    return result


def validate_no_leakage(patient_ids: Iterable[str], development_ids: Iterable[str]) -> None:
    patients, development = set(patient_ids), set(development_ids)
    for value in patients | development:
        identifier(value)
    if patients & development:
        raise ValueError("Development/test patient leakage detected")


def require_same_patients(expected: set[str], actual: set[str]) -> None:
    if expected != actual:
        raise ValueError("Patient alignment failure: missing or unexpected records")
