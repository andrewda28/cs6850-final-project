"""Uniform patient sampling, independent of source order and global RNG state."""

import random
from collections.abc import Iterable

from src.schemas import Patient, identifier
from src.validation import index_unique, validate_no_leakage


def sample_test_cohort(
    patients: Iterable[Patient], development_ids: Iterable[str], *, size: int, seed: int
) -> list[Patient]:
    by_id = index_unique(patients)
    development = set(development_ids)
    for patient_id in development:
        identifier(patient_id)
    if type(size) is not int or size < 1:
        raise ValueError("size must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    eligible = sorted(set(by_id) - development)
    if size > len(eligible):
        raise ValueError("Requested sample exceeds eligible population")
    selected = random.Random(seed).sample(eligible, size)
    validate_no_leakage(selected, development)
    return [by_id[patient_id] for patient_id in selected]
