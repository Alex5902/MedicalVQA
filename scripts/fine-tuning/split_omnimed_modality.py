#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict # For grouping by image

def main():
    parser = argparse.ArgumentParser(description="Split a specific modality from OmniMedVQA into train, validation, and test sets based on unique images.")
    parser.add_argument(
        "--input-json", type=str, required=True,
        help="Path to the full OmniMedVQA JSON file (e.g., qa_items.json)."
    )
    parser.add_argument(
        "--modality", type=str, required=True,
        help="The specific modality_type string to filter and split."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save the train.json, val.json, and test.json files for the modality."
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7,
        help="Proportion of the modality's *unique images* to use for training."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Proportion of the modality's *unique images* to use for validation."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits."
    )
    parser.add_argument(
        "--min-images", type=int, default=10, # Changed from min-samples to min-images
        help="Minimum number of unique images required for a modality to be processed."
    )

    args = parser.parse_args()

    input_json_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    target_modality = args.modality

    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1 and (args.train_ratio + args.val_ratio) < 1):
        print("Error: Train and Val ratios must be between 0 and 1, and their sum must be less than 1.", file=sys.stderr)
        sys.exit(1)

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    print(f"Splitting modality '{target_modality}' based on unique images with ratios: Train={args.train_ratio:.2f}, Val={args.val_ratio:.2f}, Test={test_ratio:.2f}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_json_path.exists():
        print(f"Error: Input JSON not found: {input_json_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading full dataset from {input_json_path}...")
    try:
        with open(input_json_path, 'r') as f:
            full_data = json.load(f)
    except Exception as e:
        print(f"Error loading input JSON: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Filtering for modality: '{target_modality}'...")
    modality_items_all_qa = [item for item in full_data if item.get("modality_type", "").lower() == target_modality.lower()]

    if not modality_items_all_qa:
        print(f"Error: No items found for modality '{target_modality}'. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(modality_items_all_qa)} QA pairs for modality '{target_modality}'.")
    print("Grouping QA pairs by unique image_path...")

    # --- Group QA pairs by image_path ---
    qa_by_image = defaultdict(list)
    for item in modality_items_all_qa:
        image_path = item.get("image_path")
        if image_path: # Ensure image_path exists
            qa_by_image[image_path].append(item)
        else:
            print(f"Warning: Item with question_id '{item.get('question_id')}' has no image_path. Skipping.", file=sys.stderr)

    unique_image_paths = list(qa_by_image.keys())
    num_unique_images = len(unique_image_paths)
    print(f"Found {num_unique_images} unique images for modality '{target_modality}'.")

    if num_unique_images < args.min_images: # Check against min_images
        print(f"Error: Modality '{target_modality}' has only {num_unique_images} unique images, which is less than the minimum required ({args.min_images}). Skipping.", file=sys.stderr)
        sys.exit(1)

    # Shuffle the list of unique image paths
    random.seed(args.seed)
    random.shuffle(unique_image_paths)

    # --- Split unique image paths ---
    # First split: separate out the test set of images
    if args.train_ratio + args.val_ratio == 0:
        print("Error: train_ratio + val_ratio cannot be zero if test set is expected.", file=sys.stderr)
        sys.exit(1)

    train_val_image_paths, test_image_paths = train_test_split(
        unique_image_paths,
        test_size=test_ratio, # Proportion for the test set of images
        random_state=args.seed,
        shuffle=False # Already shuffled
    )

    # Second split: split remaining images into train and validation
    train_image_paths = []
    val_image_paths = []
    if len(train_val_image_paths) > 0 : # Check if there's anything left to split
        if args.val_ratio > 0 and len(train_val_image_paths) > 1 : # Only split if val is desired and possible
            val_ratio_of_remainder = args.val_ratio / (args.train_ratio + args.val_ratio)
            # Handle cases where val_ratio_of_remainder might be 1.0 (if train_ratio is 0)
            # or if it results in a split that's too small (e.g., less than 1 sample for val)
            if val_ratio_of_remainder < 1.0 and (len(train_val_image_paths) * val_ratio_of_remainder >= 1):
                train_image_paths, val_image_paths = train_test_split(
                    train_val_image_paths,
                    test_size=val_ratio_of_remainder,
                    random_state=args.seed,
                    shuffle=False # Already shuffled
                )
            else: # Assign all to train if val split is not meaningful
                train_image_paths = train_val_image_paths
        else: # No validation split desired or possible
            train_image_paths = train_val_image_paths
    else: # No items left for train/val after test split (e.g. if test_ratio was 1.0)
        pass


    # --- Assign QA pairs to their respective splits based on image_path ---
    train_items = []
    for img_path in train_image_paths:
        train_items.extend(qa_by_image[img_path])

    val_items = []
    for img_path in val_image_paths:
        val_items.extend(qa_by_image[img_path])

    test_items = []
    for img_path in test_image_paths:
        test_items.extend(qa_by_image[img_path])

    # Shuffle the QA pairs within each split for good measure (optional)
    random.shuffle(train_items)
    random.shuffle(val_items)
    random.shuffle(test_items)

    print(f"\nSplit sizes for '{target_modality}' (based on QA pairs after image-level split):")
    print(f"  Train: {len(train_items)} QA pairs (from {len(train_image_paths)} unique images)")
    print(f"  Validation: {len(val_items)} QA pairs (from {len(val_image_paths)} unique images)")
    print(f"  Test: {len(test_items)} QA pairs (from {len(test_image_paths)} unique images)")
    print(f"  Total QA pairs processed: {len(train_items) + len(val_items) + len(test_items)}")
    print(f"  Total unique images processed: {len(train_image_paths) + len(val_image_paths) + len(test_image_paths)}")


    # Save the splits
    for split_name, data_list in [("train", train_items), ("val", val_items), ("test", test_items)]:
        if not data_list and split_name == "val" and args.val_ratio == 0:
            print(f"Skipping save for empty '{split_name}' set as val_ratio was 0.")
            continue
        # Warn if any split is empty (unless it's an intentionally empty val set)
        if not data_list and not (split_name == "val" and args.val_ratio == 0):
            print(f"Warning: '{split_name}' set is empty. This might be due to small dataset size and ratios.", file=sys.stderr)

        output_file = output_dir / f"{split_name}.json"
        print(f"Saving {split_name} set ({len(data_list)} QA pairs) to {output_file}...")
        try:
            with open(output_file, 'w') as f:
                json.dump(data_list, f, indent=4)
        except Exception as e:
            print(f"Error writing {output_file}: {e}", file=sys.stderr)

    print(f"\n--- Image-level splitting for modality '{target_modality}' complete ---")
    print(f"Output files are in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()

# seed=42, 123, 456 (fundus, x-ray, microscopy)