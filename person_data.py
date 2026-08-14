"""Person dataset metadata and upload matching."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PERSON_DIR = BASE_DIR / "person_data"
METADATA_PATH = PERSON_DIR / "metadata.json"
LABELED_DIR = PERSON_DIR / "labeled"


def load_persons() -> list[dict]:
    if not METADATA_PATH.exists():
        return []
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _dataset_hash(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c != "Class"]
    payload = df[cols].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _match_from_filename(filename: str) -> dict | None:
    if not filename:
        return None
    name_part = Path(filename).stem.replace("_100_transactions", "")
    slug = re.sub(r"[^a-z0-9_]", "", name_part.lower().replace(" ", "_"))
    for person in load_persons():
        if person["id"] == slug or person["name"].lower().replace(" ", "_") in slug:
            return person
        expected = person["name"].replace(" ", "_").lower()
        if expected in slug or slug in expected:
            return person
    return None


def identify_person(df: pd.DataFrame, filename: str = "") -> dict | None:
    by_name = _match_from_filename(filename)
    if by_name:
        return by_name

    upload_hash = _dataset_hash(df)
    for person in load_persons():
        if person.get("data_hash") == upload_hash:
            return person

    if len(df) == 100:
        for person in load_persons():
            labeled_path = LABELED_DIR / f"{person['id']}.csv"
            if not labeled_path.exists():
                continue
            labeled = pd.read_csv(labeled_path)
            upload_cols = [c for c in df.columns if c != "Class"]
            if list(labeled[upload_cols].columns) == list(df[upload_cols].columns):
                if labeled[upload_cols].equals(df[upload_cols].reset_index(drop=True)):
                    return person
    return None


def get_labeled_dataset(person_id: str) -> pd.DataFrame | None:
    path = LABELED_DIR / f"{person_id}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None
