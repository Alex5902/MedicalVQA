# Add matplotlib import at the top
import matplotlib.pyplot as plt
# Other imports remain the same...
import json
import numpy as np
import pandas as pd
from scipy.special import softmax as scipy_softmax
from typing import List, Dict, Tuple, Any
import ast
from tqdm import tqdm
import argparse # Already added
from pathlib import Path # Add Path import

# --- Define Adaptive Threshold Mappings ---
# Example thresholds based on your suggestions
ADAPTIVE_CONF_THRESHOLDS = {
    2: 0.6,  # Threshold for 2 options
    3: 0.5,  # Threshold for 3 options
    4: 0.4,  # Threshold for 4 options
    # Add defaults or handling for other counts if necessary
    'default': 0.35 # Example default if options != 2, 3, or 4
}

DEFAULT_MARGIN_THRESHOLD = 0.15

# --- Helper functions (parse_score_string, get_options_and_scores,
# --- calculate_softmax_probs, calculate_metrics_for_item) remain the same ---
def parse_score_string(score_str: str) -> List[float]:
    """Safely parses the string representation of a list of scores."""
    try:
        scores = ast.literal_eval(score_str)
        if isinstance(scores, list) and all(isinstance(x, (int, float)) for x in scores):
            return [float(s) for s in scores]
        else:
            # print(f"Warning: Could not parse score string correctly: {score_str}")
            return []
    except (ValueError, SyntaxError, TypeError) as e:
        # print(f"Error parsing score string '{score_str}': {e}")
        return []

def get_options_and_scores(item: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    """Extracts options and parses scores from a result dictionary item."""
    options = []
    scores_str = item.get("confidence", "[]")
    scores = parse_score_string(scores_str)
    for key in ["option_A", "option_B", "option_C", "option_D"]:
        option_text = item.get(key)
        if option_text is not None:
            options.append(option_text)
    if len(options) != len(scores):
        # print(f"Warning: Mismatch options ({len(options)}) vs scores ({len(scores)}) for item {item.get('question_id', 'N/A')}. Scores: {scores_str}")
        return [], []
    return options, scores

def calculate_softmax_probs(scores: List[float]) -> np.ndarray:
    """Converts loss scores (lower is better) to softmax probabilities."""
    if not scores: return np.array([])
    neg_scores = -np.array(scores)
    probabilities = scipy_softmax(neg_scores)
    return probabilities

def calculate_metrics_for_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates softmax probs and various accuracy metrics using adaptive thresholds."""
    metrics = {
            "question_id": item.get("question_id", "N/A"),
            "gt_answer": item.get("gt_answer"),
            "options": [], "raw_scores": [], "softmax_probs": [],
            "predicted_option_text": None, "predicted_option_index": -1,
            "is_correct_raw": False,
            "is_correct_adaptive_confidence": False,
            "is_correct_adaptive_margin": False,
            "num_options": 0,
            "effective_conf_threshold": np.nan,
            "effective_margin_threshold": np.nan,
            "top_confidence": 0.0,
            "score_margin": 0.0,
            "prob_margin": np.nan,
            "entropy": 0.0
        }
    
    options, raw_scores = get_options_and_scores(item)
    if not options or not raw_scores: return metrics

    num_options = len(options)
    metrics["num_options"] = num_options

    metrics["options"] = options
    metrics["raw_scores"] = raw_scores
    softmax_probs = calculate_softmax_probs(raw_scores)
    # Convert numpy types explicitly for broader compatibility if saving later (e.g. JSON)
    metrics["softmax_probs"] = [float(p) for p in softmax_probs] # Store as list of standard floats
    if len(softmax_probs) > 0:
        predicted_option_index = np.argmax(softmax_probs)
        metrics["predicted_option_text"] = options[predicted_option_index]
        metrics["predicted_option_index"] = int(predicted_option_index)
        top_confidence = float(softmax_probs[predicted_option_index])
        metrics["top_confidence"] = top_confidence

        is_correct_raw = (metrics["predicted_option_text"] == metrics["gt_answer"])
        metrics["is_correct_raw"] = bool(is_correct_raw)

        # <<< Apply Adaptive Confidence Threshold >>>
        effective_conf_thresh = ADAPTIVE_CONF_THRESHOLDS.get(num_options, ADAPTIVE_CONF_THRESHOLDS['default'])
        metrics["effective_conf_threshold"] = effective_conf_thresh # Store threshold used
        if top_confidence >= effective_conf_thresh:
            metrics["is_correct_adaptive_confidence"] = bool(is_correct_raw) # Use new key name
        else:
            metrics["is_correct_adaptive_confidence"] = False # Use new key name

        # <<< Apply Margin Threshold (Using fixed default here) >>>
        effective_margin_thresh = DEFAULT_MARGIN_THRESHOLD # Could be adaptive like confidence
        metrics["effective_margin_threshold"] = effective_margin_thresh # Store threshold used
        if len(raw_scores) > 1:
            sorted_scores = np.sort(raw_scores)
            score_margin = sorted_scores[1] - sorted_scores[0]
            metrics["score_margin"] = float(score_margin)
            if score_margin >= effective_margin_thresh: # Use effective threshold
                 metrics["is_correct_adaptive_margin"] = bool(is_correct_raw) # Use new key name
            else:
                 metrics["is_correct_adaptive_margin"] = False # Use new key name
        else:
             metrics["score_margin"] = float(np.inf)
             metrics["is_correct_adaptive_margin"] = bool(is_correct_raw if effective_margin_thresh <= 0 else False) # Use new key name

        if len(softmax_probs) > 1:
            sorted_probs = np.sort(softmax_probs)[::-1] # Sort descending
            prob_margin = sorted_probs[0] - sorted_probs[1]
            metrics["prob_margin"] = float(prob_margin)
        else:
            metrics["prob_margin"] = np.nan # Only one option, margin is undefined
        epsilon = 1e-9
        metrics["entropy"] = float(-np.sum(softmax_probs * np.log(softmax_probs + epsilon))) # Store as standard float
    return metrics

# --- Main Processing Function (No changes needed here) ---
def analyze_prefix_scores(results_json_path: str) -> pd.DataFrame:
    """Loads results JSON, calculates metrics for each item, and returns a DataFrame."""
    print(f"Loading results from: {results_json_path}")
    try:
        with open(results_json_path, 'r') as f:
            results_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_json_path}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {results_json_path}: {e}")
        return pd.DataFrame()

    pred_dict = results_data.get("pred_dict", [])
    if not pred_dict:
        print(f"Warning: 'pred_dict' not found or empty in {results_json_path}.")
        return pd.DataFrame()

    all_metrics = []
    print(f"Calculating metrics for {len(pred_dict)} predictions from {Path(results_json_path).name}...")
    print(f"Using Adaptive Confidence Thresholds: {ADAPTIVE_CONF_THRESHOLDS}")
    print(f"Using Fixed Margin Threshold: {DEFAULT_MARGIN_THRESHOLD}") # Adjust if margin becomes adaptive

    for item in tqdm(pred_dict, desc="Processing items"):
        metrics = calculate_metrics_for_item(item)
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df


# --- Comparison and Plotting Function (Modified to return results) ---
def compare_and_plot_confidence(real_df: pd.DataFrame,
                                dummy_df: pd.DataFrame,
                                output_plot_path: str = "confidence_drop_markers.png" # New default name
                               ) -> Tuple[pd.DataFrame, float]:
    """Merges real and dummy results, calculates drop, plots confidence markers sorted by real confidence, and returns results."""

    merged_df = pd.DataFrame() # Initialize empty df
    average_drop = np.nan # Initialize as NaN

    if real_df.empty or dummy_df.empty:
        print("Error: One or both input DataFrames are empty. Cannot compare.")
        return merged_df, average_drop

    # Select relevant columns and rename for clarity before merging
    real_conf = real_df[['question_id', 'top_confidence']].rename(columns={'top_confidence': 'top_conf_real'})
    dummy_conf = dummy_df[['question_id', 'top_confidence']].rename(columns={'top_confidence': 'top_conf_dummy'})

    # Merge based on question_id
    merged_df = pd.merge(real_conf, dummy_conf, on='question_id', how='inner')

    if merged_df.empty:
        print("Error: No matching question_ids found between real and dummy results.")
        return merged_df, average_drop

    # Calculate confidence drop
    merged_df['conf_drop'] = merged_df['top_conf_real'] - merged_df['top_conf_dummy']
    average_drop = merged_df['conf_drop'].mean()
    print(f"\nAverage drop in top confidence (Real - Dummy): {average_drop:.4f}")

    # Sort by real image confidence for plotting (still needed for the X-axis order)
    merged_df_sorted = merged_df.sort_values(by='top_conf_real', ascending=False).reset_index()

    # --- Plotting (Markers Only) ---
    plt.figure(figsize=(12, 6)) # Keep original figure size

    # Plot Real Image Confidence as points
    plt.plot(merged_df_sorted.index, merged_df_sorted['top_conf_real'],
             label='Real Image Top Confidence',
             marker='o',       # Specify marker shape (e.g., 'o' for circle)
             linestyle='None', # Specify NO line connecting markers
             markersize=3      # Adjust marker size
            )

    # Plot Dummy Image Confidence as points
    plt.plot(merged_df_sorted.index, merged_df_sorted['top_conf_dummy'],
             label='Dummy Image Top Confidence',
             marker='x',       # Use a different marker (e.g., 'x')
             linestyle='None', # Specify NO line connecting markers
             markersize=3,     # Adjust marker size
             alpha=0.7         # Keep transparency
            )

    # Add labels and title (same as before)
    plt.xlabel("Question Index (Sorted by Real Image Confidence Descending)")
    plt.ylabel("Top Softmax Probability")
    plt.title("Confidence Drop Comparison: Real vs. Dummy Images (Markers Only)") # Updated title

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0, 1.05) # Keep Y axis limit

    try:
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"Confidence comparison marker plot saved to: {output_plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close() # Close plot to free memory

    return merged_df, average_drop # Return the merged data and average drop

def plot_probability_margin_histogram(real_df: pd.DataFrame,
                                      dummy_df: pd.DataFrame,
                                      output_plot_path: str = "prob_margin_histogram.png",
                                      num_bins: int = 20):
    """Plots overlapping histograms of the softmax probability margin.""" # Updated docstring

    if real_df.empty or dummy_df.empty:
        print("Warning: One or both input DataFrames are empty. Skipping probability margin histogram.")
        return

    real_margins = real_df.loc[real_df['prob_margin'].notna(), 'prob_margin']
    dummy_margins = dummy_df.loc[dummy_df['prob_margin'].notna(), 'prob_margin']

    if real_margins.empty and dummy_margins.empty:
        print("Warning: No valid probability margins found in either dataset. Skipping histogram.")
        return

    bin_edges = np.linspace(0, 1, num_bins + 1)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    plt.hist(real_margins, bins=bin_edges, alpha=0.6, label='Real Images', density=True)
    plt.hist(dummy_margins, bins=bin_edges, alpha=0.6, label='Dummy Images', density=True)

    # <<< Update labels and title >>>
    plt.xlabel("Margin (Top Probability - Second Probability)")
    plt.ylabel("Density")
    plt.title("Distribution of Prediction Probability Margins: Real vs. Dummy Images")

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, 1) # Probability margin is between 0 and 1

    try:
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"Probability margin histogram saved to: {output_plot_path}") # Updated message
    except Exception as e:
        print(f"Error saving probability margin histogram: {e}")
    plt.close()


# --- Main Execution (Modified for Saving) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Prefix Scoring results and compare Real vs Dummy.")
    parser.add_argument("--real-results", type=str, required=True, help="Path to the JSON results file from REAL image run.")
    parser.add_argument("--dummy-results", type=str, required=True, help="Path to the JSON results file from DUMMY image run.")
    parser.add_argument("--plot-output", type=str, default="confidence_comparison_scatter.png", help="Filename for the confidence comparison plot.")
    parser.add_argument("--output-dir", type=str, default="./analysis_output", help="Directory to save analysis results (CSVs, JSON summary).")
    parser.add_argument("--real-csv", type=str, default="real_metrics_adaptive.csv", help="Filename for the real image metrics CSV.")
    parser.add_argument("--dummy-csv", type=str, default="dummy_metrics_adaptive.csv", help="Filename for the dummy image metrics CSV.")
    parser.add_argument("--compare-csv", type=str, default="comparison_metrics.csv", help="Filename for the comparison (merged) CSV.")
    parser.add_argument("--summary-json", type=str, default="summary_metrics_adaptive.json", help="Filename for the summary metrics JSON.")
    parser.add_argument("--margin-hist-output", type=str, default="margin_histogram.png", help="Filename for the margin distribution histogram plot.")

    cli_args = parser.parse_args()

    # --- Create output directory ---
    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving analysis results to: {output_dir.resolve()}")

    # --- Process REAL results (No threshold args passed) ---
    real_processed_df = analyze_prefix_scores(cli_args.real_results)

    # --- Process DUMMY results (No threshold args passed) ---
    dummy_processed_df = analyze_prefix_scores(cli_args.dummy_results)

    # --- Save individual DataFrames ---
    if not real_processed_df.empty:
        real_csv_path = output_dir / cli_args.real_csv
        try:
            real_processed_df.to_csv(real_csv_path, index=False)
            print(f"Real image metrics saved to: {real_csv_path}")
        except Exception as e:
            print(f"Error saving real metrics CSV: {e}")

    if not dummy_processed_df.empty:
        dummy_csv_path = output_dir / cli_args.dummy_csv
        try:
            dummy_processed_df.to_csv(dummy_csv_path, index=False)
            print(f"Dummy image metrics saved to: {dummy_csv_path}")
        except Exception as e:
            print(f"Error saving dummy metrics CSV: {e}")

    # --- Compare, Plot, and get Comparison Results ---
    merged_comparison_df, avg_conf_drop = compare_and_plot_confidence(
        real_processed_df,
        dummy_processed_df,
        output_dir / cli_args.plot_output # Save plot in output dir
    )

    # --- Save Comparison DataFrame ---
    if not merged_comparison_df.empty:
        compare_csv_path = output_dir / cli_args.compare_csv
        try:
            merged_comparison_df.to_csv(compare_csv_path, index=False)
            print(f"Comparison metrics saved to: {compare_csv_path}")
        except Exception as e:
            print(f"Error saving comparison CSV: {e}")


    plot_probability_margin_histogram( # Call the renamed function
        real_processed_df,
        dummy_processed_df,
        output_dir / cli_args.margin_hist_output # Argument name is fine
    )

    # --- Calculate and Save Summary Metrics ---
    summary_data = {
        "input_files": { "real": cli_args.real_results, "dummy": cli_args.dummy_results },
        "threshold_scheme": { # Describe the scheme used
            "confidence": ADAPTIVE_CONF_THRESHOLDS,
            "margin": DEFAULT_MARGIN_THRESHOLD # Or describe if adaptive
        },
        "real_image_summary": {}, "dummy_image_summary": {}, "comparison_summary": {}
    }

    if not real_processed_df.empty:
        # --- Use NEW column names for summary ---
        summary_data["real_image_summary"] = {
            "raw_accuracy": real_processed_df['is_correct_raw'].mean(),
            "adaptive_confidence_accuracy": real_processed_df['is_correct_adaptive_confidence'].mean(),
            "adaptive_margin_accuracy": real_processed_df['is_correct_adaptive_margin'].mean(),
            "average_top_confidence": real_processed_df['top_confidence'].mean(),
            "average_entropy": real_processed_df['entropy'].mean(),
            "count": len(real_processed_df)
        }
        print("\n--- REAL Image Accuracy Summary (Adaptive Thresholds) ---")
        for key, value in summary_data["real_image_summary"].items():
             print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        print("--------------------------------------------------------")


    if not dummy_processed_df.empty:
        # --- Use NEW column names for summary ---
        summary_data["dummy_image_summary"] = {
            "raw_accuracy": dummy_processed_df['is_correct_raw'].mean(),
            "adaptive_confidence_accuracy": dummy_processed_df['is_correct_adaptive_confidence'].mean(),
            "adaptive_margin_accuracy": dummy_processed_df['is_correct_adaptive_margin'].mean(),
            "average_top_confidence": dummy_processed_df['top_confidence'].mean(),
            "average_entropy": dummy_processed_df['entropy'].mean(),
            "count": len(dummy_processed_df)
        }
        print("\n--- DUMMY Image Accuracy Summary (Adaptive Thresholds) ---")
        for key, value in summary_data["dummy_image_summary"].items():
             print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        print("---------------------------------------------------------")

    # --- Comparison Summary (Unchanged) ---
    summary_data["comparison_summary"] = {
            "average_confidence_drop": avg_conf_drop if not np.isnan(avg_conf_drop) else None,
            "compared_question_count": len(merged_comparison_df) if not merged_comparison_df.empty else 0
    }
    print("\n--- Comparison Summary ---")
    print(f"Average Confidence Drop (Real - Dummy): {avg_conf_drop:.4f}" if not np.isnan(avg_conf_drop) else "N/A")
    print(f"Number of Questions Compared: {len(merged_comparison_df)}" if not merged_comparison_df.empty else 0)
    print("--------------------------")

    # --- Save Summary (Unchanged) ---
    summary_json_path = output_dir / cli_args.summary_json
    try:
        with open(summary_json_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"Summary metrics saved to: {summary_json_path}")
    except Exception as e:
        print(f"Error saving summary JSON: {e}")



# /home/alex.ia/miniconda3/envs/blip2_lavis_env/bin/python /home/alex.ia/OmniMed/scripts/analyse_scores.py \
#     --real-results "/home/alex.ia/OmniMed/results/blip2/blip2-flant5xl_real/fundus_photography/blip2/_tmp_omni_fundus_photography_fthhfrds_fundus_photography.json" \
#     --dummy-results "/home/alex.ia/OmniMed/results/blip2_dummy/blip2-flant5xl_dummy/fundus_photography/blip2/_tmp_omni_fundus_photography_8jgum_f__fundus_photography.json" \
#     --output-dir "/home/alex.ia/OmniMed/results/analysis_metrics" \
#     --real-csv "real_metrics_fundus.csv" \
#     --dummy-csv "dummy_metrics_fundus.csv" \
#     --compare-csv "comparison_metrics_fundus.csv" \
#     --summary-json "summary_metrics_fundus.json" \
#     --plot-output "confidence_drop_plot_fundus.png" \
#     --margin-hist-output "margin_histogram_fundus.png"