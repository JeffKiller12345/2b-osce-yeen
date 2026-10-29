#!/usr/bin/env python3
"""
jsonl_to_csv.py
---------------
Converts an existing station_bank.jsonl (from the old version of generate_stations.py)
into station_bank.csv, using exactly the same columns as the new generate_stations.py
and the 2b_histories Supabase table.

Usage
  python jsonl_to_csv.py                                   # station_bank.jsonl -> station_bank.csv
  python jsonl_to_csv.py --in old.jsonl --out new.csv
  python jsonl_to_csv.py --append                          # add to an existing CSV (skips ids already in it)
  python jsonl_to_csv.py --force                           # overwrite an existing CSV
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

# Must match CSV_COLUMNS in generate_stations.py and the 2b_histories table
CSV_COLUMNS = [
    "id", "created_at", "model",
    "specialty", "presentation", "presentation_common",
    "diagnosis", "diagnosis_common", "question_types",
    "patient_name", "patient_age", "patient_sex", "patient_dob",
    "candidate_info", "patient_info", "examiner_info",
    "setting", "presenting_complaint", "opening_statement", "patient_script",
    "patient_follow_up_question", "examiner_questions", "history_marks", "question_marks",
]


def js(x):
    """Lists/dicts -> JSON text (imports into jsonb columns)."""
    return json.dumps(x, ensure_ascii=False)


def to_bool(x) -> str:
    return str(bool(x)).lower()


def iso_dob(value: str) -> str:
    """Old files store DOB as dd/mm/yyyy; Postgres wants yyyy-mm-dd."""
    value = (value or "").strip()
    if not value:
        return ""
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return value  # leave as is if it is in an unexpected format


def record_to_row(rec: dict) -> dict:
    parts = rec.get("parts", {}) or {}
    return {
        "id": rec["id"],
        "created_at": rec.get("created_at", ""),
        "model": rec.get("model", ""),
        "specialty": rec["specialty"],
        "presentation": rec["presentation"],
        "presentation_common": to_bool(rec.get("presentation_common")),
        "diagnosis": rec["diagnosis"],
        "diagnosis_common": to_bool(rec.get("diagnosis_common")),
        "question_types": js(rec.get("question_types", [])),
        "patient_name": rec.get("patient_name", parts.get("patient_name", "")),
        "patient_age": rec.get("patient_age", parts.get("patient_age", "")),
        "patient_sex": str(rec.get("patient_sex", parts.get("patient_sex", ""))).strip().lower(),
        "patient_dob": iso_dob(rec.get("patient_dob", "")),
        "candidate_info": rec.get("candidate_info", ""),
        "patient_info": rec.get("patient_info", ""),
        "examiner_info": rec.get("examiner_info", ""),
        "setting": parts.get("setting", ""),
        "presenting_complaint": parts.get("presenting_complaint", ""),
        "opening_statement": parts.get("opening_statement", ""),
        "patient_script": parts.get("patient_script", ""),
        "patient_follow_up_question": parts.get("patient_follow_up_question", ""),
        "examiner_questions": js(parts.get("examiner_questions", [])),
        "history_marks": js(parts.get("history_marks", [])),
        "question_marks": js(parts.get("question_marks", [])),
    }


def existing_ids(path: Path) -> set:
    ids = set()
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                ids.add(rec["id"])
    return ids


def main():
    ap = argparse.ArgumentParser(description="Convert station_bank.jsonl to station_bank.csv")
    ap.add_argument("--in", dest="inp", default="station_bank.jsonl", help="input JSONL file")
    ap.add_argument("--out", default="station_bank.csv", help="output CSV file")
    ap.add_argument("--append", action="store_true", help="append to an existing CSV (skips ids already in it)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing CSV")
    args = ap.parse_args()

    in_path, out_path = Path(args.inp), Path(args.out)
    if not in_path.exists():
        sys.exit(f"Cannot find {in_path}.")

    out_has_data = out_path.exists() and out_path.stat().st_size > 0
    if out_has_data and not (args.append or args.force):
        sys.exit(f"{out_path} already exists. Use --append to add to it or --force to overwrite it.")

    skip_ids = existing_ids(out_path) if (out_has_data and args.append) else set()
    mode = "a" if (out_has_data and args.append) else "w"
    write_header = mode == "w"

    converted = skipped = bad = 0
    with in_path.open(encoding="utf-8") as fin, out_path.open(mode, newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec["id"] in skip_ids:
                    skipped += 1
                    continue
                writer.writerow(record_to_row(rec))
                converted += 1
            except (json.JSONDecodeError, KeyError) as e:
                bad += 1
                print(f"  line {line_no}: skipped ({type(e).__name__}: {e})")

    print(f"Done. {converted} station(s) written to {out_path}"
          + (f", {skipped} already present" if skipped else "")
          + (f", {bad} bad line(s) skipped" if bad else "")
          + ".")


if __name__ == "__main__":
    main()