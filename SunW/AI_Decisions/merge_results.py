#!/usr/bin/env python3
"""
Merge valid (non-error) experiment trials from SunW/AI_Decisions/results and archive/
into SunW/AI_Decisions/results/pvf_merged_20261025.csv.
"""

import csv
from pathlib import Path

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

def merge_csvs():
    base_dir = Path(__file__).resolve().parent / "results"
    output_path = base_dir / "pvf_merged_20261025.csv"

    # Find all relevant CSV files in results/ and archive/
    # Exclude merged files, dpo files, and any non-trial CSVs
    csv_paths = []
    for p in sorted(base_dir.rglob("*.csv")):
        # Only process pvf_*.csv and trials_*.csv
        if p.name.startswith("pvf_merged"):
            continue
        if "dpo" in p.name.lower():
            continue
        if p.name.startswith("pvf_") or p.name.startswith("trials_"):
            csv_paths.append(p)

    print(f"Found {len(csv_paths)} relevant CSV files to process:")
    for p in csv_paths:
        print(f"  - {p.relative_to(base_dir)}")

    merged_rows = []

    for path in csv_paths:
        is_trials_file = path.name.startswith("trials_")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            file_valid = 0
            for row in reader:
                # Filter for only entries where AI model did not return an error
                err = (row.get("error") or "").strip()
                if err:
                    continue

                clean_row = {}
                for col in TARGET_COLUMNS:
                    if col == "plan":
                        if is_trials_file:
                            clean_row["plan"] = "paid"
                        else:
                            clean_row["plan"] = row.get("plan", "paid")
                    else:
                        clean_row[col] = row.get(col, "")

                merged_rows.append(clean_row)
                file_valid += 1
            print(f"  -> Extracted {file_valid} valid rows from {path.name}")

    # Sort merged rows chronologically by timestamp
    merged_rows.sort(key=lambda r: r.get("timestamp", ""))

    print(f"\nTotal valid rows merged: {len(merged_rows)}")

    # Write merged output
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TARGET_COLUMNS)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Saved merged dataset to: {output_path}")

    # Print summary statistics
    from collections import Counter
    print("\nSummary by Plan:")
    for plan, count in Counter(r["plan"] for r in merged_rows).items():
        print(f"  {plan}: {count}")

    print("\nSummary by Model:")
    for model, count in Counter(r["model"] for r in merged_rows).items():
        print(f"  {model}: {count}")

if __name__ == "__main__":
    merge_csvs()
