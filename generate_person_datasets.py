"""Create 10 person datasets (100 transactions each) from creditcard.csv."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "creditcard.csv"
OUT_DIR = BASE / "person_data"
DATASET_DIR = OUT_DIR / "datasets"
LABELED_DIR = OUT_DIR / "labeled"

PERSONS = [
    {"id": "james_mitchell", "name": "James Mitchell", "city": "Chicago, IL"},
    {"id": "sarah_thompson", "name": "Sarah Thompson", "city": "Austin, TX"},
    {"id": "michael_rodriguez", "name": "Michael Rodriguez", "city": "Miami, FL"},
    {"id": "emily_chen", "name": "Emily Chen", "city": "Seattle, WA"},
    {"id": "david_washington", "name": "David Washington", "city": "Atlanta, GA"},
    {"id": "jessica_martinez", "name": "Jessica Martinez", "city": "Denver, CO"},
    {"id": "robert_anderson", "name": "Robert Anderson", "city": "Boston, MA"},
    {"id": "amanda_foster", "name": "Amanda Foster", "city": "Phoenix, AZ"},
    {"id": "christopher_lee", "name": "Christopher Lee", "city": "Portland, OR"},
    {"id": "rachel_bennett", "name": "Rachel Bennett", "city": "Nashville, TN"},
]

FRAUD_PER_PERSON = [4, 6, 3, 7, 5, 2, 8, 4, 5, 6]  # 50 fraud total across 10 people


def dataset_hash(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c != "Class"]
    payload = df[cols].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    df = pd.read_csv(SOURCE)
    fraud = df[df["Class"] == 1].sample(n=sum(FRAUD_PER_PERSON), random_state=42)
    normal = df[df["Class"] == 0].sample(n=1000 - sum(FRAUD_PER_PERSON), random_state=42)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    LABELED_DIR.mkdir(parents=True, exist_ok=True)

    metadata = []
    fraud_idx = 0
    normal_idx = 0
    fraud_list = fraud.reset_index(drop=True)
    normal_list = normal.reset_index(drop=True)

    for i, person in enumerate(PERSONS):
        n_fraud = FRAUD_PER_PERSON[i]
        n_normal = 100 - n_fraud

        chunk_fraud = fraud_list.iloc[fraud_idx : fraud_idx + n_fraud]
        chunk_normal = normal_list.iloc[normal_idx : normal_idx + n_normal]
        fraud_idx += n_fraud
        normal_idx += n_normal

        chunk = pd.concat([chunk_normal, chunk_fraud]).sample(frac=1, random_state=100 + i)
        chunk = chunk.reset_index(drop=True)

        download_name = f"{person['name'].replace(' ', '_')}_100_transactions.csv"
        upload_df = chunk.drop(columns=["Class"])
        labeled_df = chunk.copy()

        upload_path = DATASET_DIR / f"{person['id']}.csv"
        labeled_path = LABELED_DIR / f"{person['id']}.csv"
        upload_df.to_csv(upload_path, index=False)
        labeled_df.to_csv(labeled_path, index=False)

        actual_fraud = int(labeled_df["Class"].sum())
        metadata.append(
            {
                **person,
                "download_name": download_name,
                "transactions": 100,
                "actual_fraud_cases": actual_fraud,
                "actual_normal_cases": 100 - actual_fraud,
                "data_hash": dataset_hash(upload_df),
            }
        )
        print(f"{person['name']}: {actual_fraud} fraud / 100 transactions")

    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved 10 datasets to {DATASET_DIR}")


if __name__ == "__main__":
    main()
