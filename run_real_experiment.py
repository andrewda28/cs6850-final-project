import argparse
import json
from pathlib import Path

import pandas as pd

from build_icd_code_set import build_icd_code_set
from recover_icd_versions import recover_icd_versions

from src.data_loader import HUMAN_LABEL_COLUMN
from src.schemas import ReferenceLabel
from src.evaluation import evaluate
from src.llm_client import LocalConfig, create_client

from src.models import (
    ICDBaseline,
    SingleLLM,
    MultiAgent,
    AdaptiveAgent
)


def load_test_patients(diagnosis_file):

    base = Path("data/processed")

    cohort = pd.read_csv(
        base / "test_cohort.csv",
        dtype=str,
        keep_default_na=False
    )

    labels = pd.read_csv(
        base / "test_reference_labels.csv",
        dtype=str,
        keep_default_na=False
    )

    exclusions = pd.read_csv(
        base / "development_exclusions.csv",
        dtype=str
    )

    if len(cohort) != 50 or len(labels) != 50:
        raise ValueError("Expected 50 reviewed patients")

    if cohort["subject_id"].duplicated().any():
        raise ValueError("Duplicate test patients")

    if set(cohort["subject_id"]) & set(exclusions["subject_id"]):
        raise ValueError("Development data leakage")

    keys = ["subject_id", "hadm_id"]

    cohort_keys = set(map(tuple, cohort[keys].values))
    label_keys = set(map(tuple, labels[keys].values))

    if cohort_keys != label_keys:
        raise ValueError("Test cohort and labels do not match")

    if not labels["human_label"].isin(["0", "1", "-1"]).all():
        raise ValueError("Invalid human labels")

    cohort = cohort.drop(
        columns=["human_label", "reviewer", "review_seconds"],
        errors="ignore"
    )

    merged = cohort.merge(
        labels,
        on=keys,
        validate="one_to_one"
    )

    merged[HUMAN_LABEL_COLUMN] = merged["human_label"]

    output = base / "labeled_test_cohort.csv"

    merged.to_csv(output, index=False)

    patients, _ = recover_icd_versions(
        output,
        diagnosis_file
    )

    references = [
        ReferenceLabel(
            patient_id=row["subject_id"],
            label=int(row["human_label"]),
            human_review_seconds=float(row["review_seconds"])
        )
        for _, row in merged.iterrows()
        if row["human_label"] in {"0", "1"}
    ]

    if not references:
        raise ValueError("No binary reference labels")

    if {p.patient_id for p in patients} != {
        r.patient_id for r in references
    }:
        raise ValueError("Patient and reference mismatch")

    print("Selected test patients:", len(merged))
    print("Evaluable patients:", len(references))
    print("Uncertain exclusions:",
          len(merged) - len(references))

    return patients, references


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--diagnoses", required=True, type=Path)
    parser.add_argument("--dictionary", required=True, type=Path)
    parser.add_argument("--excel", required=True, type=Path)

    parser.add_argument(
        "--baseline-only",
        action="store_true"
    )

    parser.add_argument("--local-model")

    parser.add_argument(
        "--local-url",
        default="http://127.0.0.1:8000"
    )

    parser.add_argument(
        "--approved-local",
        action="store_true"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--ambiguity-lower",
        type=float,
        default=0.35
    )

    parser.add_argument(
        "--ambiguity-upper",
        type=float,
        default=0.65
    )

    args = parser.parse_args()

    if not args.baseline_only:
        if not args.local_model or not args.approved_local:
            parser.error(
                "Full evaluation requires an approved local model"
            )

    patients, references = load_test_patients(
        args.diagnoses
    )

    positive_codes = build_icd_code_set(
        args.excel,
        args.dictionary
    )

    models = [
        ICDBaseline(
            positive_codes,
            code_set_version="hw1-adrd-sd-ccw-provisional-v1"
        )
    ]

    if not args.baseline_only:

        client = create_client(
            "local",
            local_config=LocalConfig(
                model=args.local_model,
                approved=args.approved_local,
                base_url=args.local_url
            )
        )

        models.extend([
            SingleLLM(
                client,
                threshold=args.threshold
            ),
            MultiAgent(
                client,
                threshold=args.threshold
            ),
            AdaptiveAgent(
                client,
                lower=args.ambiguity_lower,
                upper=args.ambiguity_upper,
                threshold=args.threshold
            )
        ])

    results = {}

    for model in models:

        print(f"\nEvaluating {model.name}...")

        predictions = [
            model.predict(patient)
            for patient in patients
        ]

        results[model.name] = evaluate(
            references,
            predictions
        )

    report = {
        "data_version": "independent-test-cohort-v1",
        "sampling": "25 positive-enriched and 25 negative-enriched",
        "selected_patients": 50,
        "evaluated_patients": len(references),
        "uncertain_exclusions": 50 - len(references),
        "models": {
            model.name: model.describe()
            for model in models
        },
        "results": results
    }

    filename = (
        "test_icd_baseline_results.json"
        if args.baseline_only
        else "final_experiment_results.json"
    )

    output = Path("data/processed") / filename

    with open(output, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print(json.dumps(report, indent=2, allow_nan=False))
    print("\nSaved results:", output)


if __name__ == "__main__":
    main()