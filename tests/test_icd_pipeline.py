import csv
import gzip

import pandas as pd

from build_icd_code_set import build_icd_code_set
from src.schemas import DiagnosisCode


def test_build_icd_code_set(tmp_path):
    excel_file = tmp_path / "adrd_dx.xlsx"
    dictionary_file = tmp_path / "dictionary.csv"

    definitions = pd.DataFrame({
        "dx code": ["G30", "3310", "F99"],
        "code type": ["icd10", "icd9", "icd10"],
        "Notes": [
            "Alzheimer's disease",
            "Alzheimer's disease",
            "only in ccw"
        ]
    })

    definitions.to_excel(
        excel_file,
        sheet_name="adrd_sd_ccw",
        index=False
    )

    with open(dictionary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "icd_code", "icd_version", "long_title"
        ])
        writer.writerows([
            ["G30", "10", "Alzheimer's disease"],
            ["G300", "10", "Early onset"],
            ["3310", "9", "Alzheimer's disease"],
            ["F99", "10", "Excluded code"],
            ["3310", "10", "Wrong version"],
            ["Z00", "10", "Unrelated code"]
        ])

    expected = frozenset({
        DiagnosisCode("ICD-10", "G30"),
        DiagnosisCode("ICD-10", "G300"),
        DiagnosisCode("ICD-9", "3310")
    })

    assert build_icd_code_set(
        excel_file, dictionary_file
    ) == expected

    compressed_file = tmp_path / "dictionary.csv.gz"

    with open(dictionary_file, "rb") as source:
        with gzip.open(compressed_file, "wb") as target:
            target.write(source.read())

    assert build_icd_code_set(
        excel_file, compressed_file
    ) == expected