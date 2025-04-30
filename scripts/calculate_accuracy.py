#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys
from collections import defaultdict
import os

def calculate_accuracy_from_csv(csv_path: Path):
    """
    Reads a CSV file, finds the 'is_correct' column,
    and calculates the accuracy.
    Returns: tuple (total_rows, correct_rows, accuracy_percent) or None on error.
    """
    if not csv_path.is_file():
        print(f"Warning: CSV file not found at {csv_path}", file=sys.stderr)
        return None

    total_rows = 0
    correct_rows = 0
    is_correct_col_index = -1

    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            try:
                header = next(reader)
                header = [col.strip().lower() for col in header] # Lowercase for robust matching
                is_correct_col_index = header.index('is_correct')
            except StopIteration:
                print(f"Warning: CSV file '{csv_path}' is empty.", file=sys.stderr)
                return 0, 0, 0.0 # Return zero accuracy for empty file
            except ValueError:
                print(f"Error: Column 'is_correct' not found in header of {csv_path}", file=sys.stderr)
                print(f"Header found: {header}", file=sys.stderr)
                return None # Indicate error

            # Iterate over data rows
            for row_num, row in enumerate(reader, 1): # Start row count from 1 for data rows
                total_rows += 1
                try:
                    # Get value from the correct column, strip whitespace, compare case-insensitively
                    is_correct_value = row[is_correct_col_index].strip().lower()
                    if is_correct_value == 'yes':
                        correct_rows += 1
                except IndexError:
                    print(f"Warning: Row {row_num} in {csv_path.name} has fewer columns than expected. Skipping.", file=sys.stderr)
                    total_rows -= 1 # Don't count malformed row towards total
                    continue # Skip malformed rows

    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}", file=sys.stderr)
        return None # Indicate error

    if total_rows == 0:
        accuracy = 0.0
    else:
        accuracy = (correct_rows / total_rows) * 100 # Calculate percentage

    return total_rows, correct_rows, accuracy

def main():
    parser = argparse.ArgumentParser(
        description="Automatically calculate accuracy for all modalities found in a results directory."
    )
    parser.add_argument(
        "results_dir",
        help="Path to the main results directory containing modality subdirectories."
    )
    args = parser.parse_args()

    results_root_dir = Path(args.results_dir).expanduser().resolve()

    if not results_root_dir.is_dir():
        print(f"Error: Results directory not found at {results_root_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning for results in: {results_root_dir}")

    # Dictionary to store results: results[modality][metric] = (total, correct, accuracy)
    results_data = defaultdict(dict)
    modalities_found = []

    # Iterate through subdirectories in the results root
    for modality_dir in sorted(results_root_dir.iterdir()):
        if modality_dir.is_dir():
            modality_name = modality_dir.name # Use the directory name as the modality identifier
            modalities_found.append(modality_name)
            print(f"\nProcessing modality: {modality_name}")

            # Define the expected CSV filenames
            prefix_csv_path = modality_dir / "Prefix_based_Score.csv"
            qa_csv_path = modality_dir / "Question-answering_Score.csv"

            # Calculate accuracy for Prefix Score
            print(f" -> Checking {prefix_csv_path.name}...")
            prefix_result = calculate_accuracy_from_csv(prefix_csv_path)
            if prefix_result is not None:
                results_data[modality_name]["Prefix"] = prefix_result
            else:
                 results_data[modality_name]["Prefix"] = None # Mark as missing/error

            # Calculate accuracy for QA Score
            print(f" -> Checking {qa_csv_path.name}...")
            qa_result = calculate_accuracy_from_csv(qa_csv_path)
            if qa_result is not None:
                 results_data[modality_name]["QA"] = qa_result
            else:
                 results_data[modality_name]["QA"] = None # Mark as missing/error

    # --- Print Summary Table ---
    print("\n\n--- Accuracy Summary ---")
    if not results_data:
        print("No modality results found.")
        sys.exit(0)

    # Determine column widths
    max_mod_len = max(len(mod) for mod in modalities_found) if modalities_found else 10
    header_fmt = f"{{:<{max_mod_len}}} | {{:>12}} | {{:>12}}"
    row_fmt    = f"{{:<{max_mod_len}}} | {{:>12}} | {{:>12}}"

    print(header_fmt.format("Modality", "Prefix Acc %", "QA Acc %"))
    print("-" * (max_mod_len + 3 + 12 + 3 + 12)) # Separator line

    for modality in sorted(results_data.keys()): # Sort modalities alphabetically
        prefix_acc_str = "N/A"
        qa_acc_str = "N/A"

        if results_data[modality].get("Prefix") is not None:
            _, _, prefix_acc = results_data[modality]["Prefix"]
            prefix_acc_str = f"{prefix_acc:.2f}"

        if results_data[modality].get("QA") is not None:
            _, _, qa_acc = results_data[modality]["QA"]
            qa_acc_str = f"{qa_acc:.2f}"

        print(row_fmt.format(modality, prefix_acc_str, qa_acc_str))

    print("-" * (max_mod_len + 3 + 12 + 3 + 12))

if __name__ == "__main__":
    main()