#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
from collections import Counter # Import Counter

def main():
    parser = argparse.ArgumentParser(description="List unique modality_type values and counts from OmniMedVQA qa_items.json")
    parser.add_argument("json_file", help="Path to the qa_items.json file")
    args = parser.parse_args()

    json_path = Path(args.json_file).expanduser().resolve()

    if not json_path.is_file():
        print(f"Error: File not found at {json_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading modalities from: {json_path}")

    try:
        with open(json_path, 'r') as f:
            qa_items = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from {json_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read file {json_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Total items loaded: {len(qa_items)}")

    # --- Count items per modality ---
    print("Counting items per modality...")
    # Create a list of all modality types found in the data
    # Use .get() to handle potential missing keys gracefully, defaulting to "unknown"
    all_modalities_list = [item.get("modality_type", "unknown") for item in qa_items]

    # Use Counter to efficiently count occurrences of each modality
    modality_counts = Counter(all_modalities_list)
    # --- End Counting ---

    # Get unique modalities and sort them alphabetically (from the Counter's keys)
    sorted_modalities = sorted(modality_counts.keys())

    print("\nFound unique modalities and counts (sorted):")
    total_items_counted = 0
    for mod in sorted_modalities:
        count = modality_counts[mod] # Get the count for this modality
        print(f"- {mod}: {count}")
        total_items_counted += count

    # Optional: Print in a format suitable for shell array pasting
    print("\nBash array format (copy this into your SLURM script):")
    # Ensure proper quoting for modalities with spaces or special characters
    modalities_str = " ".join([f'"{mod}"' for mod in sorted_modalities])
    print(f"ALL_MODALITIES=({modalities_str})")

    print(f"\nTotal unique modalities: {len(sorted_modalities)}")
    # Sanity check: compare total counted items with original list length
    print(f"Total items counted across modalities: {total_items_counted}")
    if len(qa_items) != total_items_counted:
        print("WARNING: Total items mismatch - check data or script logic.", file=sys.stderr)


if __name__ == "__main__":
    main()
