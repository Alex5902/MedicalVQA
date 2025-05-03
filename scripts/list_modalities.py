#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
from copy import deepcopy # Use deepcopy for safety

def main():
    parser = argparse.ArgumentParser(
        description="Create modality-specific JSON files with image paths replaced by a dummy path."
    )
    parser.add_argument(
        "input_json_file",
        help="Path to the original OmniMedVQA qa_items.json file"
    )
    parser.add_argument(
        "--modalities",
        nargs='+', # Accepts one or more modality names
        required=True,
        help="List of modality names to process (e.g., 'Fundus Photography' 'X-Ray')"
    )
    parser.add_argument(
        "--dummy-image-path",
        required=True,
        help="The exact path to the pre-generated dummy black image file (e.g., 'Images/DUMMY/dummy_black_336.png')"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the new dummy JSON files will be saved."
    )
    args = parser.parse_args()

    input_json_path = Path(args.input_json_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    dummy_path = args.dummy_image_path # Keep as string provided by user

    if not input_json_path.is_file():
        print(f"Error: Input JSON file not found at {input_json_path}", file=sys.stderr)
        sys.exit(1)

    # Ensure the dummy image path isn't obviously wrong (doesn't guarantee it exists!)
    # A more robust check would try os.path.exists, but might fail on cluster paths
    if not dummy_path or len(dummy_path) < 5:
         print(f"Warning: Dummy image path '{dummy_path}' seems very short or invalid.", file=sys.stderr)
         # Consider adding a check if it's an absolute path if needed

    print(f"Reading data from: {input_json_path}")
    try:
        with open(input_json_path, 'r') as f:
            qa_items = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing {input_json_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Total items loaded: {len(qa_items)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Using dummy image path: {dummy_path}")

    processed_count = 0
    for target_modality in args.modalities:
        print(f"\nProcessing modality: '{target_modality}'")

        # Filter items for the current modality
        modality_items_original = [
            item for item in qa_items
            if item.get("modality_type", "unknown").lower() == target_modality.lower()
        ]

        if not modality_items_original:
            print(f"  No items found for modality '{target_modality}'. Skipping.")
            continue

        print(f"  Found {len(modality_items_original)} items.")

        # Create new list with modified paths
        modality_items_modified = []
        for item in modality_items_original:
            new_item = deepcopy(item) # Create a distinct copy
            original_path = new_item.get("image_path", "N/A")
            new_item["image_path"] = dummy_path # Replace the path
            modality_items_modified.append(new_item)
            # Optional: print a sample change
            # if len(modality_items_modified) < 3:
            #    print(f"    Original path: {original_path} -> New path: {new_item['image_path']}")


        # Create output filename
        modality_slug = target_modality.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-","").replace(".","")
        output_filename = f"dummy_{modality_slug}.json"
        output_path = output_dir / output_filename

        print(f"  Writing {len(modality_items_modified)} modified items to: {output_path}")
        try:
            with open(output_path, 'w') as f:
                json.dump(modality_items_modified, f, indent=2) # Indent for readability
            processed_count += 1
            print("  Write successful.")
        except Exception as e:
            print(f"  Error writing to {output_path}: {e}", file=sys.stderr)

    print(f"\nFinished processing. Created {processed_count} dummy JSON files in {output_dir}")

if __name__ == "__main__":
    main()  