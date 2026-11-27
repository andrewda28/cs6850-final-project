"""Read the inspected HW1 export format without guessing diagnosis versions.

The user confirmed the manual annotation column is human-reviewed. Notebook code
joins all_icd_codes with '|', but discards icd_version. Model inputs therefore
require separately resolved diagnosis systems from the approved source data.
"""

import csv
from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from src.schemas import DiagnosisCode, Patient, ReferenceLabel


HUMAN_LABEL_COLUMN = (
    "is AD/ADRD? (type in 1,0, -1) 1 for yes ADRD Present, 0 for No ADRD, -1 uncertain)"
)
REQUIRED_COLUMNS = frozenset({"subject_id", "hadm_id", "text", "all_icd_codes", HUMAN_LABEL_COLUMN})


@dataclass(frozen=True, repr=False)
class _HW1Record:
    patient_id: str
    admission_id: str
    clinical_text: str
    unversioned_codes: tuple[str, ...]
    human_label: int | None


@dataclass(frozen=True)
class HW1Development:
    """Restricted in-memory data. Exclusions cover only the supplied exports.

    All teammate exports must be loaded before treating development_ids as the
    complete team exclusion set. Uncertain and unlabeled people remain excluded.
    """

    _records: tuple[_HW1Record, ...] = field(repr=False)

    @property
    def development_ids(self) -> frozenset[str]:
        return frozenset(r.patient_id for r in self._records)

    @property
    def references(self) -> tuple[ReferenceLabel, ...]:
        # Review time is unavailable in this export, so it remains unknown.
        return tuple(ReferenceLabel(r.patient_id, r.human_label) for r in self._records
                     if r.human_label in (0, 1))

    @property
    def uncertain_ids(self) -> frozenset[str]:
        return frozenset(r.patient_id for r in self._records if r.human_label == -1)

    @property
    def unlabeled_ids(self) -> frozenset[str]:
        return frozenset(r.patient_id for r in self._records if r.human_label is None)

    def to_patients(
        self, diagnoses_by_encounter: Mapping[tuple[str, str], tuple[DiagnosisCode, ...]]
    ) -> tuple[Patient, ...]:
        """Build label-free model inputs after source-based diagnosis resolution.

        Keys are (subject_id, hadm_id). The caller must resolve code systems from
        the original diagnosis source, using the same admission as the export.
        Code sets must match the exported lists exactly; no version is inferred
        from spelling and no missing mapping becomes an empty/negative diagnosis.
        Uncertain/unlabeled patients remain present; references contains only the
        binary-reviewed subset and evaluation will reject unresolved labels.
        """
        patients = []
        for record in self._records:
            codes = diagnoses_by_encounter.get((record.patient_id, record.admission_id))
            if not isinstance(codes, tuple) or not codes or not all(isinstance(c, DiagnosisCode) for c in codes):
                raise ValueError("Missing or invalid versioned diagnoses for an HW1 encounter")
            if {c.code for c in codes} != set(record.unversioned_codes):
                raise ValueError("Versioned diagnoses do not match the exported HW1 encounter codes")
            patients.append(Patient(record.patient_id, codes, record.clinical_text))
        return tuple(patients)


def _read_rows(path: str | Path) -> Generator[dict[str, str], None, None]:
    # Discharge notes can exceed csv's default field limit. Restore the process
    # setting after this sequential read, including on errors or early closure.
    previous_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(max(previous_limit, 16 * 1024 * 1024))
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, strict=True)
            headers = reader.fieldnames
            if not headers or len(headers) != len(set(headers)) or not REQUIRED_COLUMNS <= set(headers):
                raise ValueError("HW1 export has missing or duplicate column headers")
            count = 0
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("HW1 export contains a malformed row")
                count += 1
                yield row
            if not count:
                raise ValueError("HW1 export contains no patient records")
    except (OSError, UnicodeError, csv.Error):
        raise ValueError("Unable to read HW1 CSV export") from None
    finally:
        csv.field_size_limit(previous_limit)


def load_hw1(paths: Iterable[str | Path]) -> HW1Development:
    """Combine reviewed HW1 development exports; never execute their notebooks.

    Duplicates across or within exports are errors requiring explicit resolution,
    even when annotations agree. Only the confirmed manual annotation column is
    read as truth. ICD-derived flags, rationale, and other review fields are never
    appended to the discharge text. Blank and -1 labels are kept unresolved.
    """
    if isinstance(paths, (str, Path)):
        raise ValueError("Provide a collection of HW1 export paths")
    records, seen = [], set()
    for path in paths:
        rows = _read_rows(path)
        try:
            for row in rows:
                patient_id, admission_id = row["subject_id"].strip(), row["hadm_id"].strip()
                if not patient_id or not admission_id:
                    raise ValueError("HW1 export has a missing patient or admission identifier")
                if patient_id in seen:
                    raise ValueError("Duplicate patient across HW1 development records; resolve explicitly")
                clinical_text = row["text"]
                if not clinical_text.strip():
                    raise ValueError("HW1 export has missing discharge text")
                codes = tuple(code.strip() for code in row["all_icd_codes"].split("|"))
                if not all(codes):
                    raise ValueError("HW1 export has an empty or malformed diagnosis list")
                label = row[HUMAN_LABEL_COLUMN].strip()
                labels = {"": None, "0": 0, "1": 1, "-1": -1, "0.0": 0, "1.0": 1, "-1.0": -1}
                if label not in labels:
                    raise ValueError("HW1 human labels must be 0, 1, -1 (uncertain), or blank")
                records.append(_HW1Record(patient_id, admission_id, clinical_text, codes, labels[label]))
                seen.add(patient_id)
        finally:
            rows.close()
    if not records:
        raise ValueError("Provide at least one nonempty HW1 export")
    return HW1Development(tuple(sorted(records, key=lambda r: r.patient_id)))
