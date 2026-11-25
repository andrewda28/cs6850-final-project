"""HW1 integration boundary; no assumptions about restricted source schemas."""

from src.schemas import Patient, ReferenceLabel


def load_hw1() -> tuple[list[Patient], list[ReferenceLabel], set[str]]:
    """Implement in the approved data environment after reviewing the HW1 notebook.

    Resolve stable patient identity, encounter aggregation, observation windows,
    diagnosis systems, text selection, reviewed labels, review effort, and the
    complete development-patient exclusion list. Do not copy raw data into Git.
    """
    raise NotImplementedError("HW1 notebook and approved data mapping are required")
