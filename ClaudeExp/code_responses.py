#!/usr/bin/env python3
"""Add Decision and Length columns to an Ollama experiment CSV.

Usage:
    python3 code_responses.py results/trials_20261003_120305.csv

The coded file is written next to the source as
trials_20261003_120305_coded.csv. Pass --in-place to replace the source file.
"""

import argparse
import csv
import re
from pathlib import Path


DECISION_WORDS = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
WORDS = re.compile(r"\b[\w']+\b")


def classify_response(response: str) -> int:
    """Classify the first explicit yes/no answer; otherwise return indeterminate."""
    match = DECISION_WORDS.search(response)
    if match is None:
        return 0
    return 1 if match.group(1).lower() == "yes" else -1


def count_words(response: str) -> int:
    """Count words, treating contractions such as "don't" as one word."""
    return len(WORDS.findall(response))


def output_path(source: Path, in_place: bool) -> Path:
    if in_place:
        return source
    return source.with_name(f"{source.stem}_coded{source.suffix}")


def code_csv(source: Path, destination: Path) -> int:
    with source.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or "response" not in reader.fieldnames:
            raise ValueError("The input CSV must contain a 'response' column.")

        fieldnames = [
            field for field in reader.fieldnames if field not in {"Decision", "Length"}
        ] + ["Decision", "Length"]

        rows_written = 0
        with destination.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                response = row.get("response") or ""
                row["Decision"] = classify_response(response)
                row["Length"] = count_words(response)
                writer.writerow({field: row.get(field, "") for field in fieldnames})
                rows_written += 1

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add Decision and Length columns to an experiment CSV."
    )
    parser.add_argument("csv_file", type=Path, help="Experiment CSV to code.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the input CSV instead of writing a _coded.csv file.",
    )
    args = parser.parse_args()

    source = args.csv_file.resolve()
    if not source.is_file():
        parser.error(f"CSV file not found: {source}")

    destination = output_path(source, args.in_place)
    if args.in_place:
        temporary_destination = source.with_suffix(".tmp")
        rows_written = code_csv(source, temporary_destination)
        temporary_destination.replace(source)
    else:
        rows_written = code_csv(source, destination)

    print(f"Wrote {rows_written} coded rows to {destination}")


if __name__ == "__main__":
    main()