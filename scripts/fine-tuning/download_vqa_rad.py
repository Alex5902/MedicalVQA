#!/usr/bin/env python3
"""
Downloads the VQA-RAD dataset (images + QA pairs) from Hugging Face Hub
using the datasets library.
"""
import argparse
import sys
from pathlib import Path
import shutil
from datasets import load_dataset, DatasetDict
from PIL import Image as PIL_Image # Import Pillow's Image module and alias it
import json # Import the json module globally
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Download VQA-RAD from Hugging Face Hub.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./finetuning_datasets/VQA_RAD",
        help="Directory to save the dataset images and JSON metadata."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_images_dir = output_dir / "images"
    output_json_path = output_dir / "vqa_rad_train.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "flaviagiammarino/vqa-rad"
    print(f"Attempting to download dataset '{repo_id}' from Hugging Face Hub...")
    print(f"Saving to: {output_dir.resolve()}")

    try:
        # Load the dataset
        dataset = load_dataset(repo_id)

        if isinstance(dataset, DatasetDict):
            ds_split = dataset.get("train", dataset[list(dataset.keys())[0]])
        else:
            ds_split = dataset

        print(f"Dataset loaded. Number of examples: {len(ds_split)}")
        print("Features:", ds_split.features)

        processed_data = []
        print(f"Copying/saving images to {output_images_dir} and preparing JSON data...")

        for i, example in enumerate(tqdm(ds_split, desc="Processing examples")):
            try:
                if 'image' not in example or example['image'] is None:
                    print(f"Warning: Missing 'image' field or image data in example {i}. Skipping.", file=sys.stderr)
                    continue

                img_pil = example['image'] # This is a PIL.Image.Image object

                # --- Filename generation ---
                img_filename_base = None
                original_path_str = getattr(img_pil, 'path', None) # HF datasets often stores path here

                if 'question_id' in example and example['question_id']:
                    img_filename_base = str(example['question_id']).replace('/', '_').replace(' ', '_') # Sanitize
                elif 'qid' in example and example['qid']:
                    img_filename_base = str(example['qid']).replace('/', '_').replace(' ', '_') # Sanitize
                elif original_path_str: # Try to derive from original path if no ID
                    img_filename_base = Path(original_path_str).stem
                else: # Last resort: use index
                    img_filename_base = f"vqarad_image_{i:04d}"
                # --- End Filename generation ---

                img_filename = None
                dest_path = None

                if original_path_str and Path(original_path_str).exists():
                    source_path = Path(original_path_str)
                    # Use original suffix if available, else guess from PIL format or default to .png
                    suffix = source_path.suffix if source_path.suffix else f".{img_pil.format.lower() if img_pil.format else 'png'}"
                    img_filename = f"{img_filename_base}{suffix}"
                    dest_path = output_images_dir / img_filename
                    shutil.copy2(source_path, dest_path)
                elif isinstance(img_pil, PIL_Image.Image): # Check against PIL_Image.Image
                    img_format = img_pil.format if img_pil.format else 'PNG' # Default to PNG
                    img_filename = f"{img_filename_base}.{img_format.lower()}"
                    dest_path = output_images_dir / img_filename
                    img_pil.save(dest_path)
                else:
                    print(f"Warning: Image data for example {i} (ID: {img_filename_base}) is not a PIL Image or path. Type: {type(img_pil)}. Skipping.", file=sys.stderr)
                    continue

                record = {
                    "image_path": f"images/{img_filename}", # Path relative to output JSON
                    "question": example.get("question"),
                    "answer": example.get("answer"),
                    "question_id": example.get("question_id") or example.get("qid") or img_filename_base,
                    "question_type": example.get("question_type"),
                    "answer_type": example.get("answer_type"),
                }
                record = {k: v for k, v in record.items() if v is not None}
                processed_data.append(record)

            except Exception as e:
                print(f"Error processing example {i} (attempted ID: {img_filename_base}): {e}. Skipping.", file=sys.stderr)
                # print("Failed example data:", example) # Uncomment for more debug info


        if not processed_data:
            print("Warning: No data was processed successfully. Output JSON will be empty.", file=sys.stderr)

        print(f"\nSaving metadata for {len(processed_data)} examples to {output_json_path}...")
        with open(output_json_path, 'w') as f:
            json.dump(processed_data, f, indent=4) # 'json' is now globally defined

        print("\n--- VQA-RAD Download & Preparation Complete ---")
        print(f"Images saved in: {output_images_dir.resolve()}")
        print(f"Metadata JSON saved to: {output_json_path.resolve()}")

    except Exception as e:
        print(f"\nError during dataset loading or overall processing: {e}", file=sys.stderr)
        print("Please ensure the 'datasets' library is installed (`pip install datasets pillow`) and you have internet access.")
        sys.exit(1)

if __name__ == "__main__":
    main()