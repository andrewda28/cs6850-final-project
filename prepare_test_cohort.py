import argparse
from pathlib import Path

import pandas as pd


def prepare_test_cohort(
    positive_file,
    negative_file,
    exclusion_file,
    size_per_group=25,
    seed=6850
):

    if size_per_group < 1:
        raise ValueError("Sample size must be positive")

    positive = pd.read_csv(positive_file, dtype=str)
    negative = pd.read_csv(negative_file, dtype=str)

    exclusions = pd.read_csv(
        exclusion_file,
        dtype=str
    )

    excluded_ids = set(exclusions["subject_id"])

    required = {
        "subject_id",
        "hadm_id",
        "text",
        "all_icd_codes"
    }

    for df in [positive, negative]:

        if not required.issubset(df.columns):
            raise ValueError("Missing required columns")

        if df["subject_id"].isna().any():
            raise ValueError("Missing patient identifiers")

        if df["subject_id"].duplicated().any():
            raise ValueError("Duplicate patient identifiers")

    # Remove development patients
    positive = positive[
        ~positive["subject_id"].isin(excluded_ids)
    ].copy()

    negative = negative[
        ~negative["subject_id"].isin(excluded_ids)
    ].copy()

    # Remove patients appearing in both source groups
    overlap = (
        set(positive["subject_id"])
        & set(negative["subject_id"])
    )

    positive = positive[
        ~positive["subject_id"].isin(overlap)
    ]

    negative = negative[
        ~negative["subject_id"].isin(overlap)
    ]

    # Require usable clinical notes and diagnoses
    def valid_records(df):

        mask = pd.Series(True, index=df.index)

        for column in required:
            mask &= (
                df[column].fillna("").str.strip() != ""
            )

        return df[mask]

    positive = valid_records(positive)
    negative = valid_records(negative)

    print("Eligible positive candidates:", len(positive))
    print("Eligible negative candidates:", len(negative))
    print("Cross-group overlaps excluded:", len(overlap))

    if min(len(positive), len(negative)) < size_per_group:
        raise ValueError("Insufficient eligible patients")

    # Reproducible sampling
    positive_sample = positive.sort_values(
        "subject_id"
    ).sample(n=size_per_group, random_state=seed)

    negative_sample = negative.sort_values(
        "subject_id"
    ).sample(n=size_per_group, random_state=seed + 1)

    positive_sample = positive_sample.assign(
        sampling_group="positive_enriched"
    )

    negative_sample = negative_sample.assign(
        sampling_group="negative_enriched"
    )

    cohort = pd.concat(
        [positive_sample, negative_sample],
        ignore_index=True
    )

    cohort = cohort.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    # Validate no leakage
    if set(cohort["subject_id"]) & excluded_ids:
        raise ValueError("Development data leakage")

    if cohort["subject_id"].duplicated().any():
        raise ValueError("Duplicate test patients")

    cohort["human_label"] = ""
    cohort["reviewer"] = ""
    cohort["review_seconds"] = ""

    print("\nTEST COHORT PREPARED")
    print("Total patients:", len(cohort))
    print("Positive enriched:", size_per_group)
    print("Negative enriched:", size_per_group)
    print("Development overlap: 0")
    print("Duplicate patients: 0")

    return cohort


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--positive", required=True, type=Path
    )

    parser.add_argument(
        "--negative", required=True, type=Path
    )

    parser.add_argument(
        "--size-per-group", type=int, default=25
    )

    parser.add_argument(
        "--seed", type=int, default=6850
    )

    args = parser.parse_args()

    cohort = prepare_test_cohort(
        args.positive,
        args.negative,
        Path("data/processed/development_exclusions.csv"),
        args.size_per_group,
        args.seed
    )

    output = Path("data/processed/test_cohort.csv")

    cohort.to_csv(output, index=False)

    print("Saved cohort:", output)


if __name__ == "__main__":
    main()