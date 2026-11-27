import csv
import gzip
import pandas as pd

from pathlib import Path
from collections import Counter
from src.schemas import DiagnosisCode


def build_icd_code_set(excel_file, dictionary_file):

    df = pd.read_excel(
        excel_file,
        sheet_name="adrd_sd_ccw",
        dtype=str
    )

    df = df.loc[df["Notes"] != "only in ccw"]

    prefixes = {"9": set(), "10": set()}

    for _, row in df.iterrows():

        code = row["dx code"]
        code_type = row["code type"]

        if pd.isna(code) or pd.isna(code_type):
            raise ValueError("Missing code definition")

        code = code.strip()
        version = code_type.strip().lower().replace("icd", "")

        if version not in prefixes or not code:
            raise ValueError("Invalid ICD definition")

        prefixes[version].add(code)

    positive_codes = set()
    matched_prefixes = set()

    open_file = (
        gzip.open
        if str(dictionary_file).endswith(".gz")
        else open
    )

    with open_file(
        dictionary_file, "rt", newline="", encoding="utf-8"
    ) as f:

        for row in csv.DictReader(f):

            code = (row.get("icd_code") or "").strip()
            version = (row.get("icd_version") or "").strip()

            if not code or version not in {"9", "10"}:
                raise ValueError("Missing or invalid ICD dictionary entry")

            for prefix in prefixes.get(version, set()):
                if code.startswith(prefix):

                    positive_codes.add(
                        DiagnosisCode(
                            system=f"ICD-{version}",
                            code=code
                        )
                    )

                    matched_prefixes.add((version, prefix))

    counts = Counter(c.system for c in positive_codes)

    unmatched = {
        (version, prefix)
        for version, codes in prefixes.items()
        for prefix in codes
    } - matched_prefixes

    print("\nICD CODE SET BUILT")
    print("Original prefixes:", sum(map(len, prefixes.values())))
    print("Expanded ICD-9 codes:", counts["ICD-9"])
    print("Expanded ICD-10 codes:", counts["ICD-10"])
    print("Total expanded codes:", len(positive_codes))
    print("Unmatched prefixes:", sorted(unmatched))

    if not positive_codes:
        raise ValueError("No matching ICD codes found")

    return frozenset(positive_codes)


if __name__ == "__main__":

    base = Path.home() / "Desktop/SYSEN5650/data_hf_ad"

    codes = build_icd_code_set(
        base / "adrd_dx.xlsx",
        base / "mimic-iv-3.1/hosp/d_icd_diagnoses.csv"
    )

    for code in sorted(codes, key=lambda c: (c.system, c.code)):
        print(code.system, code.code)