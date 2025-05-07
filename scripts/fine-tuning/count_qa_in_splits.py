import json
from pathlib import Path
import argparse

def count_qa_pairs(json_file_path: Path) -> int:
    """Loads a JSON file and returns the number of items in the top-level list."""
    if not json_file_path.exists():
        print(f"Error: File not found - {json_file_path}")
        return -1 # Indicate error
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        else:
            print(f"Error: JSON content in {json_file_path} is not a list.")
            return -2 # Indicate wrong format
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}.")
        return -3 # Indicate JSON error
    except Exception as e:
        print(f"An unexpected error occurred with {json_file_path}: {e}")
        return -4

def main():
    parser = argparse.ArgumentParser(description="Count QA pairs in split JSON files for OmniMedVQA modalities.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./finetuning_datasets/OmniMedVQA_Splits_ImgLevel",
        help="Base directory where modality subdirectories with train/val/test.json files are located."
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs='+', # Allows multiple modality names
        required=True,
        help="List of modality subdirectory names (e.g., 'Fundus_Photography' 'X-Ray' 'Microscopy_Images')."
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    print(f"Checking splits in base directory: {base_dir.resolve()}")

    for modality_name in args.modalities:
        print(f"\n--- Modality: {modality_name} ---")
        modality_dir = base_dir / modality_name
        if not modality_dir.is_dir():
            print(f"  Warning: Directory not found for modality {modality_name} at {modality_dir}")
            continue

        for split_type in ["train", "val", "test"]:
            json_file = modality_dir / f"{split_type}.json"
            count = count_qa_pairs(json_file)
            if count >= 0:
                print(f"  {split_type.capitalize()} set ({json_file.name}): {count} QA pairs")
            else:
                print(f"  {split_type.capitalize()} set ({json_file.name}): Error or file not found (code: {count})")

if __name__ == "__main__":
    main()