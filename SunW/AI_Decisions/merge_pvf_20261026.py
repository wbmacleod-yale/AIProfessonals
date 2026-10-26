#!/usr/bin/env python3
"""
Merge SunW/AI_Decisions/results/pvf_merged_20261025.csv and
SunW/AI_Decisions/results/pvf_20261026_160948.csv into
SunW/AI_Decisions/results/pvf_merged_20261026.csv, keeping only entries
where the AI model did not return an error.
"""

import csv
from pathlib import Path
from collections import Counter

TARGET_COLUMNS = [
    "plan",
    "provider",
    "model",
    "object",
    "judge_role",
    "rule_given",
    "trial",
    "prompt",
    "decision",
    "mentions_hart",
    "response",
    "timestamp",
    "latency_seconds",
    "input_tokens",
    "output_tokens",
    "error",
]

def merge_pvf_20261026():
    base_dir = Path(__file__).resolve().parent / "results"
    file_prev = base_dir / "pvf_merged_20261025.csv"
    file_new = base_dir / "pvf_20261026_160948.csv"
    output_path = base_dir / "pvf_merged_20261026.csv"

    print(f"Reading previous merged file: {file_prev.name}")
    print(f"Reading new experiment file:  {file_new.name}")

    merged_rows = []
    seen_timestamps = set()

    for path in [file_prev, file_new]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        valid_count = 0
        error_count = 0
        duplicate_count = 0

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check for error
                err = (row.get("error") or "").strip()
                if err:
                    error_count += 1
                    continue

                # Ensure non-empty response/decision
                if not (row.get("decision") or "").strip():
                    error_count += 1
                    continue

                ts = (row.get("timestamp") or "").strip()
                if ts in seen_timestamps:
                    duplicate_count += 1
                    continue
                if ts:
                    seen_timestamps.add(ts)

                clean_row = {}
                for col in TARGET_COLUMNS:
                    val = row.get(col, "")
                    if col == "plan":
                        val = (val or "paid").strip().lower()
                    clean_row[col] = val

                merged_rows.append(clean_row)
                valid_count += 1

        print(f"  {path.name}: {valid_count} valid rows kept, {error_count} error rows excluded, {duplicate_count} duplicates skipped.")

    # Sort merged rows chronologically by timestamp
    merged_rows.sort(key=lambda r: r.get("timestamp", ""))

    print(f"\nTotal valid rows in merged dataset: {len(merged_rows)}")

    # Write output
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TARGET_COLUMNS)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Saved merged dataset to: {output_path}")

    # Summary by Plan and Model
    print("\nBreakdown by Plan:")
    for plan, count in Counter(r["plan"] for r in merged_rows).items():
        print(f"  {plan}: {count}")

    print("\nBreakdown by Model:")
    for model, count in Counter(r["model"] for r in merged_rows).items():
        print(f"  {model}: {count}")

    print("\nBreakdown by Plan & Model:")
    for (plan, model), count in Counter((r["plan"], r["model"]) for r in merged_rows).items():
        print(f"  {plan} | {model}: {count}")

if __name__ == "__main__":
    merge_pvf_20261026()
