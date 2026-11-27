import pandas as pd
import time
from pathlib import Path
import shutil

path = Path("data/processed/review_assignments/conrad_review.csv")

df = pd.read_csv(path, dtype=str, keep_default_na=False)

backup = path.with_suffix(".csv.bak")
if not backup.exists():
    shutil.copy2(path, backup)

for i, row in df.iterrows():

    if (
        row["human_label"] in ["0", "1", "-1"]
        and row["reasoning"].strip()
        and row["review_seconds"].strip()
    ):
        continue

    print("\n" + "=" * 70)
    print(f"PATIENT {i + 1} OF {len(df)}")
    print("Review ID:", row["review_id"])
    print("Age:", row["admit_age"])
    print("=" * 70)

    start = time.monotonic()

    print("\nCLINICAL NOTES:\n")
    print(row["text"])

    print("\nRADIOLOGY NOTES:\n")
    print(row["radiology_notes"])

    print("\nMEDICATIONS:")

    for med in [
        "denepezil", "memantine", "tacrine",
        "rivastigmine", "galantamine"
    ]:
        print(f"{med}: {row[med]}")

    print("\n1 = AD/ADRD present")
    print("0 = AD/ADRD not identified")
    print("-1 = Uncertain")
    print("q = Quit and resume later")

    while True:
        label = input("\nEnter label: ").strip()

        if label.lower() == "q":
            print("Previous reviews saved.")
            raise SystemExit

        if label in ["1", "0", "-1"]:
            break

        print("Please enter 1, 0, -1, or q.")

    reasoning = ""
    while not reasoning:
        reasoning = input("Brief reasoning: ").strip()

    seconds = round(time.monotonic() - start)

    df.at[i, "human_label"] = label
    df.at[i, "reasoning"] = reasoning
    df.at[i, "review_seconds"] = str(seconds)

    temp = path.with_suffix(".tmp")
    df.to_csv(temp, index=False)
    temp.replace(path)

    print(f"\nReview saved! Time: {seconds} seconds.")

print("\nALL REVIEWS COMPLETE")
print(df["human_label"].value_counts())
