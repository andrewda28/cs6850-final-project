
import csv
from pathlib import Path
from collections import defaultdict, Counter
from src.data_loader import load_hw1, HUMAN_LABEL_COLUMN

csv.field_size_limit(16 * 1024 * 1024)

INPUT_DIR = Path("data/hw1")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "merged_hw1.csv"

files = sorted(INPUT_DIR.glob("*.csv"))

patients = defaultdict(list)
fieldnames = None

for file in files:
    with file.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, strict=True)

        required_columns = {
            "subject_id",
            "hadm_id",
            "text",
            "all_icd_codes",
            HUMAN_LABEL_COLUMN
        }

        if not required_columns.issubset(reader.fieldnames):
            raise ValueError("Missing required CSV columns")

        if fieldnames is None:
            fieldnames = list(reader.fieldnames)
        else:
            for column in reader.fieldnames:
                if column not in fieldnames:
                    fieldnames.append(column)

        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed CSV record")

            patient_id = row["subject_id"].strip()

            if not patient_id:
                raise ValueError("Missing patient ID")

            patients[patient_id].append((file.name, row))

if not patients:
    raise ValueError("No patient records found")

merged = []
audit = []
duplicate_groups = 0
conflicts = 0

for patient_id, records in sorted(patients.items()):
    sources = [source for source, row in records]
    rows = [row for source, row in records]

    if len(rows) > 1:
        duplicate_groups += 1

    admissions = {row["hadm_id"].strip() for row in rows}

    if len(admissions) != 1:
        raise ValueError("Duplicate patient has different admissions")

    texts = {row["text"].strip() for row in rows}

    diagnoses = {
        frozenset(code.strip() for code in
                  row["all_icd_codes"].split("|"))
        for row in rows
    }

    if len(texts) != 1 or len(diagnoses) != 1:
        raise ValueError("Duplicate patient has different clinical data")

    label_map = {
        "": "",
        "1": "1",
        "1.0": "1",
        "0": "0",
        "0.0": "0",
        "-1": "-1",
        "-1.0": "-1"
    }

    labels = [
        label_map[row[HUMAN_LABEL_COLUMN].strip()]
        for row in rows
    ]

    if len(set(labels)) == 1:
        final_label = labels[0]
    else:
        final_label = "-1"
        conflicts += 1

    selected = rows[0].copy()
    selected[HUMAN_LABEL_COLUMN] = final_label

    merged.append(selected)

    if len(rows) > 1:
        for source, label in zip(sources, labels):
            audit.append({
                "subject_id": patient_id,
                "source": source,
                "original_label": label,
                "merged_label": final_label
            })

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(merged)

with (OUTPUT_DIR / "duplicate_audit.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "subject_id",
            "source",
            "original_label",
            "merged_label"
        ]
    )
    writer.writeheader()
    writer.writerows(audit)

cohort = load_hw1([OUTPUT_FILE])
counts = Counter(r.label for r in cohort.references)

print("\nMERGE SUCCESSFUL")
print("CSV files:", len(files))
print("Original records:", sum(len(v) for v in patients.values()))
print("Unique patients:", len(merged))
print("Duplicate groups:", duplicate_groups)
print("Duplicates removed:", sum(len(v)-1 for v in patients.values()))
print("Annotation conflicts:", conflicts)
print("Confirmed positive:", counts[1])
print("Confirmed negative:", counts[0])
print("Uncertain:", len(cohort.uncertain_ids))
print("Unlabeled:", len(cohort.unlabeled_ids))
print("Binary-labeled patients:", len(cohort.references))
print("Output:", OUTPUT_FILE)
