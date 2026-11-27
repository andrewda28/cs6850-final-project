from pathlib import Path

import pandas as pd


def main():

    cohort = pd.read_csv(
        "data/processed/test_cohort.csv",
        dtype=str
    )

    reviewers = [
        "conrad",
        "andrew",
        "gabriella",
        "lina",
        "xingyu"
    ]

    if len(cohort) != 50:
        raise ValueError("Expected 50 patients")

    if cohort["subject_id"].duplicated().any():
        raise ValueError("Duplicate patients")

    cohort["review_id"] = [
        f"TEST-{i + 1:03d}"
        for i in range(len(cohort))
    ]

    output = Path("data/processed/review_assignments")
    output.mkdir(parents=True, exist_ok=True)

    review_columns = [
        "review_id",
        "admit_age",
        "text",
        "radiology_notes",
        "denepezil",
        "memantine",
        "tacrine",
        "rivastigmine",
        "galantamine"
    ]

    assignments = []

    for i, reviewer in enumerate(reviewers):

        subset = cohort.iloc[
            i * 10:(i + 1) * 10
        ].copy()

        review = subset[review_columns].copy()

        review["human_label"] = ""
        review["reasoning"] = ""
        review["review_seconds"] = ""

        review.to_csv(
            output / f"{reviewer}_review.csv",
            index=False
        )

        mapping = subset[
            ["review_id", "subject_id", "hadm_id"]
        ].copy()

        mapping["reviewer"] = reviewer
        assignments.append(mapping)

        print(f"{reviewer}: {len(review)} patients")

    pd.concat(
        assignments,
        ignore_index=True
    ).to_csv(
        output / "assignment_mapping.csv",
        index=False
    )

    print("\nREVIEW ASSIGNMENTS GENERATED")


if __name__ == "__main__":
    main()