from pathlib import Path
from math import isfinite

import pandas as pd


def collect_review_labels(review_dir):

    review_dir = Path(review_dir)

    mapping = pd.read_csv(
        review_dir / "assignment_mapping.csv",
        dtype=str,
        keep_default_na=False
    )

    if len(mapping) != 50:
        raise ValueError("Expected 50 assigned patients")

    if mapping["review_id"].duplicated().any():
        raise ValueError("Duplicate review IDs")

    if mapping["subject_id"].duplicated().any():
        raise ValueError("Duplicate patients")

    completed = []

    for reviewer in mapping["reviewer"].unique():

        reviews = pd.read_csv(
            review_dir / f"{reviewer}_review.csv",
            dtype=str,
            keep_default_na=False
        )

        expected = mapping[
            mapping["reviewer"] == reviewer
        ]

        if reviews["review_id"].duplicated().any():
            raise ValueError("Duplicate completed reviews")

        if set(reviews["review_id"]) != set(expected["review_id"]):
            raise ValueError(
                f"Assignment mismatch for {reviewer}"
            )

        for _, row in reviews.iterrows():

            label = row["human_label"].strip()

            if label not in {"0", "1", "-1"}:
                raise ValueError(
                    f"Incomplete or invalid label for {reviewer}"
                )

            if not row["reasoning"].strip():
                raise ValueError(
                    f"Missing reasoning for {reviewer}"
                )

            seconds = float(row["review_seconds"])

            if not isfinite(seconds) or seconds < 0:
                raise ValueError("Invalid review time")

            completed.append({
                "review_id": row["review_id"],
                "human_label": int(label),
                "review_seconds": seconds
            })

        print(f"{reviewer}: {len(reviews)} completed reviews")

    labels = mapping.merge(
        pd.DataFrame(completed),
        on="review_id",
        validate="one_to_one"
    )

    print("\nHUMAN LABEL COLLECTION COMPLETE")
    print("Total reviewed:", len(labels))
    print("Positive:", (labels["human_label"] == 1).sum())
    print("Negative:", (labels["human_label"] == 0).sum())
    print("Uncertain:", (labels["human_label"] == -1).sum())

    return labels


def main():

    labels = collect_review_labels(
        Path("data/processed/review_assignments")
    )

    output = Path(
        "data/processed/test_reference_labels.csv"
    )

    labels.to_csv(output, index=False)

    print("Saved labels:", output)


if __name__ == "__main__":
    main()