"""HW1-shaped fixtures are entirely invented; never read actual HW1 files here."""

import csv

import pytest

from src.data_loader import HUMAN_LABEL_COLUMN, load_hw1
from src.evaluation import evaluate
from src.sampling import sample_test_cohort
from src.schemas import DiagnosisCode, Patient, Prediction


def synthetic_row(patient_id="synthetic-p", admission_id="synthetic-a", label="1"):
    return {
        "subject_id": patient_id, "hadm_id": admission_id,
        "text": "Invented discharge fixture.\nNo real patient information.",
        "all_icd_codes": "DEMO_A|DEMO_B", HUMAN_LABEL_COLUMN: label,
        "adrd_dx(icd_code)": "0", "adrd_dx(final)": "0", "reason_final": "annotation-only fixture",
    }


def write_export(tmp_path, rows, name="synthetic.csv"):
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def diagnoses(patient_id="synthetic-p", admission_id="synthetic-a"):
    return {(patient_id, admission_id): (DiagnosisCode("SYNTH", "DEMO_A"), DiagnosisCode("SYNTH", "DEMO_B"))}


def test_manual_labels_are_separate_and_missing_review_time_is_unknown(tmp_path):
    cohort = load_hw1([write_export(tmp_path, [synthetic_row()])])
    assert cohort.references[0].label == 1  # Disagrees with both fixture ICD/final fields.
    assert cohort.references[0].human_review_seconds is None
    patient, = cohort.to_patients(diagnoses())
    assert patient.clinical_text == synthetic_row()["text"]
    assert "annotation-only" not in patient.clinical_text
    assert not hasattr(patient, "human_label")
    assert "synthetic-p" not in repr(cohort)


def test_teammate_files_include_uncertain_and_unlabeled_patients_in_exclusions(tmp_path):
    first = write_export(tmp_path, [synthetic_row("synthetic-b", label="1"),
                                    synthetic_row("synthetic-a", label="-1")], "first.csv")
    second = write_export(tmp_path, [synthetic_row("synthetic-c", label=""),
                                     synthetic_row("synthetic-d", label="0")], "second.csv")
    cohort = load_hw1([first, second])
    assert cohort.development_ids == {"synthetic-a", "synthetic-b", "synthetic-c", "synthetic-d"}
    assert cohort.uncertain_ids == {"synthetic-a"}
    assert cohort.unlabeled_ids == {"synthetic-c"}
    assert [r.patient_id for r in cohort.references] == ["synthetic-b", "synthetic-d"]
    assert cohort == load_hw1([second, first])
    population = [Patient(p) for p in cohort.development_ids] + [Patient("synthetic-new")]
    assert sample_test_cohort(population, cohort.development_ids, size=1, seed=0)[0].patient_id == "synthetic-new"


@pytest.mark.parametrize("across_files", [False, True])
def test_duplicate_patients_require_resolution_even_across_admissions(tmp_path, across_files):
    first, second = synthetic_row(), synthetic_row(admission_id="synthetic-other-admission")
    paths = ([write_export(tmp_path, [first], "a.csv"), write_export(tmp_path, [second], "b.csv")]
             if across_files else [write_export(tmp_path, [first, second])])
    with pytest.raises(ValueError, match="Duplicate patient") as error:
        load_hw1(paths)
    assert "synthetic-p" not in str(error.value)


@pytest.mark.parametrize("field,value", [
    ("subject_id", ""), ("hadm_id", " "), ("text", "\n"),
    ("all_icd_codes", ""), ("all_icd_codes", "DEMO_A||DEMO_B"),
    (HUMAN_LABEL_COLUMN, "2"), (HUMAN_LABEL_COLUMN, "true"), (HUMAN_LABEL_COLUMN, "NaN"),
])
def test_invalid_fields_fail_without_echoing_values(tmp_path, field, value):
    row = synthetic_row()
    row[field] = value
    with pytest.raises(ValueError):
        load_hw1([write_export(tmp_path, [row])])


@pytest.mark.parametrize("label,expected", [("0.0", 0), ("1.0", 1), ("-1.0", None)])
def test_numeric_csv_label_formats(tmp_path, label, expected):
    cohort = load_hw1([write_export(tmp_path, [synthetic_row(label=label)])])
    if expected is None:
        assert not cohort.references
        assert cohort.uncertain_ids == {"synthetic-p"}
    else:
        assert cohort.references[0].label == expected


def test_missing_versions_cannot_silently_become_negative_diagnoses(tmp_path):
    cohort = load_hw1([write_export(tmp_path, [synthetic_row()])])
    for mapping in [{}, diagnoses(admission_id="synthetic-wrong-admission"),
                    {("synthetic-p", "synthetic-a"): ()},
                    {("synthetic-p", "synthetic-a"): (DiagnosisCode("SYNTH", "OTHER"),)}]:
        with pytest.raises(ValueError):
            cohort.to_patients(mapping)


def test_uncertain_case_remains_in_inputs_and_blocks_full_cohort_evaluation(tmp_path):
    cohort = load_hw1([write_export(tmp_path, [synthetic_row(), synthetic_row("synthetic-u", label="-1")])])
    patients = cohort.to_patients(diagnoses() | diagnoses("synthetic-u"))
    assert len(patients) == 2 and len(cohort.references) == 1
    predictions = [Prediction(p.patient_id, "synthetic-model", 0.5, 1, 0) for p in patients]
    with pytest.raises(ValueError, match="alignment"):
        evaluate(cohort.references, predictions)


def test_long_multiline_notes_and_csv_limit_restoration(tmp_path):
    row = synthetic_row()
    row["text"] = "Synthetic text, with commas and newlines.\n" * 5000
    previous_limit = csv.field_size_limit()
    cohort = load_hw1([write_export(tmp_path, [row])])
    assert cohort.to_patients(diagnoses())[0].clinical_text == row["text"]
    assert csv.field_size_limit() == previous_limit
    row[HUMAN_LABEL_COLUMN] = "invalid"
    with pytest.raises(ValueError):
        load_hw1([write_export(tmp_path, [row])])
    assert csv.field_size_limit() == previous_limit


@pytest.mark.parametrize("contents", ["", "subject_id,text\nsynthetic-p,invented\n",
                                      "subject_id,subject_id\nsynthetic-p,synthetic-p\n"])
def test_missing_or_duplicate_headers(tmp_path, contents):
    path = tmp_path / "synthetic.csv"
    path.write_text(contents)
    with pytest.raises(ValueError, match="headers"):
        load_hw1([path])


def test_empty_missing_and_malformed_files(tmp_path):
    with pytest.raises(ValueError):
        load_hw1([])
    with pytest.raises(ValueError):
        load_hw1("synthetic.csv")
    with pytest.raises(ValueError, match="Unable to read"):
        load_hw1([tmp_path / "nonexistent.csv"])
    path = tmp_path / "synthetic.csv"
    with path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=list(synthetic_row())).writeheader()
    header = path.read_text()
    with pytest.raises(ValueError, match="no patient"):
        load_hw1([path])
    path.write_text(header + "synthetic-p\n")
    with pytest.raises(ValueError, match="malformed"):
        load_hw1([path])
