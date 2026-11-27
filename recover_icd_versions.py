import argparse
import csv
import gzip

from pathlib import Path
from collections import defaultdict, Counter

from src.data_loader import load_hw1
from src.schemas import DiagnosisCode


MERGED_FILE = Path("data/processed/merged_hw1.csv")


def recover_icd_versions(merged_file, diagnosis_file):

    csv.field_size_limit(
        max(csv.field_size_limit(), 16 * 1024 * 1024)
    )

    cohort = load_hw1([merged_file])

    # Identify the encounters we need
    encounters = set()

    with open(merged_file, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (
                row["subject_id"].strip(),
                row["hadm_id"].strip()
            )
            encounters.add(key)

    print("Encounters to match:", len(encounters))

    # Recover diagnosis codes and versions
    diagnoses = defaultdict(set)

    with gzip.open(
        diagnosis_file, "rt", newline="", encoding="utf-8"
    ) as f:

        for row in csv.DictReader(f):

            key = (
                row["subject_id"].strip(),
                row["hadm_id"].strip()
            )

            if key not in encounters:
                continue

            version = row["icd_version"].strip()
            code = row["icd_code"].strip()

            if version not in {"9", "10"} or not code:
                raise ValueError("Invalid source diagnosis")

            diagnoses[key].add(
                DiagnosisCode(
                    system=f"ICD-{version}",
                    code=code
                )
            )

    # Convert diagnosis sets into tuples
    diagnoses_by_encounter = {
        key: tuple(
            sorted(codes, key=lambda c: (c.system, c.code))
        )
        for key, codes in diagnoses.items()
    }

    print("Matched encounters:", len(diagnoses_by_encounter))
    print("Missing encounters:",
          len(encounters - diagnoses.keys()))

    # Validate using Andrew's existing data loader
    patients = cohort.to_patients(diagnoses_by_encounter)

    # Keep binary-labeled patients for evaluation
    reference_ids = {
        reference.patient_id
        for reference in cohort.references
    }

    evaluation_patients = tuple(
        patient for patient in patients
        if patient.patient_id in reference_ids
    )

    version_counts = Counter(
        diagnosis.system
        for patient in patients
        for diagnosis in patient.diagnoses
    )

    print("\nICD VERSION RECOVERY SUCCESSFUL")
    print("Total patients:", len(patients))
    print("Binary-labeled patients:", len(evaluation_patients))
    print("Uncertain patients:", len(cohort.uncertain_ids))
    print("ICD-9 diagnoses:", version_counts["ICD-9"])
    print("ICD-10 diagnoses:", version_counts["ICD-10"])

    return evaluation_patients, cohort.references


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--diagnoses",
        required=True,
        type=Path
    )

    args = parser.parse_args()

    recover_icd_versions(MERGED_FILE, args.diagnoses)