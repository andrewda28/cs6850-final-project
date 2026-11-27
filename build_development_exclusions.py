import argparse
import csv

from pathlib import Path


def build_development_exclusions(paths):

    csv.field_size_limit(
        max(csv.field_size_limit(), 16 * 1024 * 1024)
    )

    development_ids = set()
    total_records = 0

    for path in paths:

        file_ids = set()

        with open(path, newline="", encoding="utf-8-sig") as f:

            reader = csv.DictReader(f)

            if "subject_id" not in (reader.fieldnames or []):
                raise ValueError("Missing subject_id column")

            for row in reader:

                patient_id = (row.get("subject_id") or "").strip()

                if not patient_id:
                    raise ValueError("Missing patient ID")

                file_ids.add(patient_id)
                total_records += 1

        development_ids.update(file_ids)

        print(f"{path.name}: {len(file_ids)} unique patients")

    print("\nDEVELOPMENT EXCLUSION SET")
    print("Records inspected:", total_records)
    print("Unique patients to exclude:", len(development_ids))

    return development_ids


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original",
        required=True,
        type=Path
    )

    args = parser.parse_args()

    paths = sorted(Path("data/hw1").glob("*.csv"))

    if len(paths) != 5:
        raise ValueError("Expected five HW1 datasets")

    paths.append(args.original)

    development_ids = build_development_exclusions(paths)

    output = Path("data/processed/development_exclusions.csv")

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["subject_id"])

        for patient_id in sorted(development_ids):
            writer.writerow([patient_id])

    print("Saved exclusion set:", output)


if __name__ == "__main__":
    main()