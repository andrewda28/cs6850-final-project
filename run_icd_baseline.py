import argparse
import json

from pathlib import Path

from build_icd_code_set import build_icd_code_set
from recover_icd_versions import recover_icd_versions

from src.models.icd_baseline import ICDBaseline
from src.evaluation import evaluate


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--diagnoses", required=True, type=Path)
    parser.add_argument("--dictionary", required=True, type=Path)
    parser.add_argument("--excel", required=True, type=Path)

    args = parser.parse_args()

    # Load patients and human reference labels
    patients, references = recover_icd_versions(
        Path("data/processed/merged_hw1.csv"),
        args.diagnoses
    )

    # Build the original HW1 ICD code set
    positive_codes = build_icd_code_set(
        args.excel,
        args.dictionary
    )

    # Initialize the ICD baseline
    model = ICDBaseline(
        positive_codes=positive_codes,
        code_set_version="hw1-adrd-sd-ccw-provisional-v1"
    )

    # Generate predictions
    predictions = [
        model.predict(patient)
        for patient in patients
    ]

    # Evaluate against human labels
    results = evaluate(references, predictions)

    print("\nICD BASELINE DEVELOPMENT RESULTS")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()