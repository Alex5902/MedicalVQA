#!/usr/bin/env python3
import json
from pathlib import Path
import argparse
import sys
from tqdm import tqdm # For progress on dummy creation

def combine_json_lists(input_files: list[Path], output_file: Path):
    """Reads multiple JSON files containing lists and combines them into one list."""
    combined_data = []
    print(f"Combining lists from {len(input_files)} files...")
    for file_path in input_files:
        if not file_path.exists():
            print(f"Warning: Input file not found: {file_path}. Skipping.", file=sys.stderr)
            continue
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                print(f"  Read {len(data)} items from {file_path.name}")
                combined_data.extend(data)
            else:
                print(f"Warning: Content of {file_path.name} is not a list. Skipping.", file=sys.stderr)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Skipping.", file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred with {file_path}: {e}. Skipping.", file=sys.stderr)

    print(f"Total items combined: {len(combined_data)}")
    print(f"Saving combined list to: {output_file}...")
    try:
        with open(output_file, 'w') as f:
            json.dump(combined_data, f, indent=4)
        print("Combined file saved successfully.")
    except Exception as e:
        print(f"Error writing combined file {output_file}: {e}", file=sys.stderr)

def create_dummy_version(input_file: Path, output_file: Path, dummy_image_path: str):
    """Creates a dummy version of a QA JSON file by replacing image paths."""
    if not input_file.exists():
        print(f"Error: Input file for dummy creation not found: {input_file}", file=sys.stderr)
        return

    print(f"\nCreating dummy version from: {input_file}")
    print(f"Using dummy image path: '{dummy_image_path}'")
    print(f"Saving dummy version to: {output_file}...")

    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file {input_file} for dummy creation: {e}", file=sys.stderr)
        return

    if not isinstance(data, list):
        print(f"Error: Input file {input_file} does not contain a list.", file=sys.stderr)
        return

    modified_data = []
    for item in tqdm(data, desc="Processing items for dummy"):
        if isinstance(item, dict):
            new_item = item.copy() # Avoid modifying original list in memory
            if "image_path" in new_item:
                new_item["image_path"] = dummy_image_path
            modified_data.append(new_item)
        else:
            print(f"Warning: Skipping non-dictionary item found in list: {item}", file=sys.stderr)
            modified_data.append(item) # Append as is? Or skip? Let's append.

    try:
        with open(output_file, 'w') as f:
            json.dump(modified_data, f, indent=4)
        print("Dummy file saved successfully.")
    except Exception as e:
        print(f"Error writing dummy file {output_file}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Combine OmniMedVQA modality test splits and create a dummy version.")
    parser.add_argument(
        "--base-split-dir", type=str, required=True,
        help="Base directory containing the modality subdirectories (e.g., ./finetuning_datasets/OmniMedVQA_Splits_ImgLevel)."
    )
    parser.add_argument(
        "--modalities", type=str, nargs='+', required=True,
        help="List of modality subdirectory names to combine (e.g., 'Fundus_Photography' 'X-Ray' 'Microscopy_Images')."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save the combined_test.json and combined_test_dummy.json files."
    )
    parser.add_argument(
        "--combined-filename", type=str, default="combined_test.json",
        help="Filename for the combined test set."
    )
    parser.add_argument(
        "--dummy-filename", type=str, default="combined_test_dummy.json",
        help="Filename for the dummy version of the combined test set."
    )
    parser.add_argument(
        "--dummy-image-path", type=str, default="Images/dummy/dummy_black_336.png", # Assumed relative path
        help="The relative image path to use for the dummy file."
    )

    args = parser.parse_args()

    base_split_dir = Path(args.base_split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    input_test_files = []
    for modality in args.modalities:
        test_file = base_split_dir / modality / "test.json"
        if test_file.exists():
            input_test_files.append(test_file)
        else:
            print(f"Warning: test.json not found for modality '{modality}' at {test_file}. Skipping.", file=sys.stderr)

    if not input_test_files:
        print("Error: No input test.json files found for the specified modalities. Exiting.", file=sys.stderr)
        sys.exit(1)

    combined_output_file = output_dir / args.combined_filename
    dummy_output_file = output_dir / args.dummy_filename

    # Step 1: Combine the test files
    combine_json_lists(input_test_files, combined_output_file)

    # Step 2: Create the dummy version from the combined file
    create_dummy_version(combined_output_file, dummy_output_file, args.dummy_image_path)

    print("\n--- Combination and Dummy Creation Complete ---")

if __name__ == "__main__":
    main()

# python ./scripts/fine-tuning/combine_test_splits.py \
#     --base-split-dir ./finetuning_datasets/OmniMedVQA_Splits_ImgLevel \
#     --modalities "Fundus_Photography" "X-Ray" "Microscopy_Images" \
#     --output-dir ./finetuning_datasets/OmniMedVQA_CombinedTest \
#     --dummy-image-path "Images/dummy/dummy_black_336.png" # Or your preferred dummy path