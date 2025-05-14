# Add matplotlib import at the top
import matplotlib.pyplot as plt
# Other imports remain the same...
import json
import numpy as np
import pandas as pd
from scipy.special import softmax as scipy_softmax # Keeping this as per your request
from typing import List, Dict, Tuple, Any
import ast
from tqdm import tqdm
import argparse
from pathlib import Path

# --- Define Adaptive Threshold Mappings ---
ADAPTIVE_CONF_THRESHOLDS = {
    2: 0.6,
    3: 0.5,
    4: 0.4,
    'default': 0.35
}
DEFAULT_MARGIN_THRESHOLD = 0.15 # This is for the existing "adaptive_margin_accuracy"

# --- Helper functions ---
def parse_score_string(score_str: str) -> List[float]:
    """Safely parses the string representation of a list of scores."""
    try:
        scores = ast.literal_eval(score_str)
        if isinstance(scores, list) and all(isinstance(x, (int, float)) for x in scores):
            return [float(s) for s in scores]
        return []
    except (ValueError, SyntaxError, TypeError):
        return []

def get_options_and_scores(item: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    """Extracts options and parses raw loss scores from a result dictionary item."""
    options = []
    # Use the field that contains raw NLL scores
    raw_scores_str = item.get("confidence_scores_neg_log_likelihood", item.get("confidence", "[]"))
    raw_scores = parse_score_string(raw_scores_str)
    
    for key in ["option_A", "option_B", "option_C", "option_D"]:
        option_text = item.get(key)
        if option_text is not None:
            options.append(str(option_text)) # Ensure options are strings

    if len(options) != len(raw_scores):
        # print(f"Warning: Mismatch options ({len(options)}) vs raw_scores ({len(raw_scores)}) for item {item.get('question_id', 'N/A')}. Raw Scores: {raw_scores_str}")
        return [], [] # Return empty if mismatch to avoid errors later
    return options, raw_scores

def calculate_softmax_probs(raw_loss_scores: List[float]) -> np.ndarray:
    """Converts raw loss scores (lower is better) to softmax probabilities."""
    if not raw_loss_scores: return np.array([])
    # Negate because lower loss = higher probability for softmax
    neg_loss_scores = -np.array(raw_loss_scores) 
    probabilities = scipy_softmax(neg_loss_scores)
    return probabilities

def calculate_metrics_for_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates softmax probs and various accuracy metrics using adaptive thresholds."""
    metrics = {
        "question_id": item.get("question_id", "N/A"),
        "gt_answer": str(item.get("gt_answer")), # Ensure GT is string
        "options": [], 
        "raw_scores": [], # Will store raw NLL scores
        "softmax_probs": [],
        "predicted_option_text": None, 
        "predicted_option_index": -1,
        "is_correct_raw": False,
        "is_correct_adaptive_confidence": False,
        "is_correct_adaptive_margin": False, # Based on raw score margin
        "num_options": 0,
        "effective_conf_threshold": np.nan,
        "effective_margin_threshold": np.nan, # For raw score margin
        "top_confidence": 0.0, # This will be top softmax probability
        "score_margin": 0.0, # Margin from raw scores
        "prob_margin": np.nan, # <<< NEW: Difference between top 2 softmax probabilities
        "entropy": 0.0
    }
    
    options, raw_scores = get_options_and_scores(item)
    if not options or not raw_scores: 
        return metrics

    num_options = len(options)
    metrics["num_options"] = num_options
    metrics["options"] = options
    metrics["raw_scores"] = [float(rs) for rs in raw_scores]

    # Calculate softmax probabilities from the raw_scores
    # This assumes your "confidence" or "confidence_scores_neg_log_likelihood" are raw NLL scores
    softmax_probs_np = calculate_softmax_probs(raw_scores)
    metrics["softmax_probs"] = [float(p) for p in softmax_probs_np]

    if len(softmax_probs_np) > 0:
        predicted_option_index = np.argmax(softmax_probs_np)
        metrics["predicted_option_text"] = options[predicted_option_index]
        metrics["predicted_option_index"] = int(predicted_option_index)
        top_confidence = float(softmax_probs_np[predicted_option_index])
        metrics["top_confidence"] = top_confidence

        is_correct_raw = (metrics["predicted_option_text"] == metrics["gt_answer"])
        metrics["is_correct_raw"] = bool(is_correct_raw)

        effective_conf_thresh = ADAPTIVE_CONF_THRESHOLDS.get(num_options, ADAPTIVE_CONF_THRESHOLDS['default'])
        metrics["effective_conf_threshold"] = effective_conf_thresh
        metrics["is_correct_adaptive_confidence"] = bool(is_correct_raw and top_confidence >= effective_conf_thresh)

        # Original score_margin based on raw scores
        effective_margin_thresh_raw_score = DEFAULT_MARGIN_THRESHOLD 
        metrics["effective_margin_threshold"] = effective_margin_thresh_raw_score
        if len(raw_scores) > 1:
            # Assuming lower raw_score is better
            sorted_raw_scores = np.sort(raw_scores) # Ascending sort
            score_margin_val = sorted_raw_scores[1] - sorted_raw_scores[0] # Margin between two best (lowest) scores
            metrics["score_margin"] = float(score_margin_val)
            metrics["is_correct_adaptive_margin"] = bool(is_correct_raw and score_margin_val >= effective_margin_thresh_raw_score)
        else:
            metrics["score_margin"] = float(np.inf)
            metrics["is_correct_adaptive_margin"] = bool(is_correct_raw and (effective_margin_thresh_raw_score <= 0 if num_options == 1 else False))

        # Calculate probability margin (top_prob - second_top_prob)
        if num_options > 1:
            sorted_probs = np.sort(softmax_probs_np)[::-1] # Sort probabilities descending
            prob_margin_val = float(sorted_probs[0] - sorted_probs[1])
            metrics["prob_margin"] = prob_margin_val
        else:
            metrics["prob_margin"] = np.nan 
        
        epsilon = 1e-9
        metrics["entropy"] = float(-np.sum(softmax_probs_np * np.log(softmax_probs_np + epsilon)))
    
    return metrics

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

    # Use "predictions" key as per your fine-tuned script's output
    pred_list = results_data.get("predictions", results_data.get("pred_dict", []))
    if not pred_list:
        print(f"Warning: 'predictions' or 'pred_dict' not found or empty in {results_json_path}.")
        return pd.DataFrame()

    all_metrics_data = []
    print(f"Calculating metrics for {len(pred_list)} predictions from {Path(results_json_path).name}...")
    print(f"Using Adaptive Confidence Thresholds: {ADAPTIVE_CONF_THRESHOLDS}")
    print(f"Using Fixed Margin Threshold (for raw score margin based accuracy): {DEFAULT_MARGIN_THRESHOLD}")

    for item in tqdm(pred_list, desc="Processing items"):
        metrics = calculate_metrics_for_item(item)
        all_metrics_data.append(metrics)

    metrics_df = pd.DataFrame(all_metrics_data)
    return metrics_df

# --- NpEncoder class for handling NaN in JSON dump ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj): # Convert float NaN to None (JSON null)
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj): 
            return None
        return super(NpEncoder, self).default(obj)

# --- Plotting functions (compare_and_plot_confidence, plot_probability_margin_histogram, plot_accuracy_vs_confidence) ---
# --- These should remain the same as your provided version. I'll omit them here for brevity but assume they are present. ---
def compare_and_plot_confidence(real_df: pd.DataFrame,
                                dummy_df: pd.DataFrame,
                                output_plot_path: str = "confidence_drop_markers.png"
                               ) -> Tuple[pd.DataFrame, float]:
    merged_df = pd.DataFrame() 
    average_drop = np.nan 
    if real_df.empty or dummy_df.empty:
        print("Error: One or both input DataFrames are empty. Cannot compare.")
        return merged_df, average_drop
    real_conf = real_df[['question_id', 'top_confidence']].rename(columns={'top_confidence': 'top_conf_real'})
    dummy_conf = dummy_df[['question_id', 'top_confidence']].rename(columns={'top_confidence': 'top_conf_dummy'})
    merged_df = pd.merge(real_conf, dummy_conf, on='question_id', how='inner')
    if merged_df.empty:
        print("Error: No matching question_ids found between real and dummy results.")
        return merged_df, average_drop
    merged_df['conf_drop'] = merged_df['top_conf_real'] - merged_df['top_conf_dummy']
    average_drop = merged_df['conf_drop'].mean()
    print(f"\nAverage drop in top confidence (Real - Dummy): {average_drop:.4f}")
    merged_df_sorted = merged_df.sort_values(by='top_conf_real', ascending=False).reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(merged_df_sorted.index, merged_df_sorted['top_conf_real'],
             label='Real Image Top Confidence', marker='o', linestyle='None', markersize=3)
    plt.plot(merged_df_sorted.index, merged_df_sorted['top_conf_dummy'],
             label='Dummy Image Top Confidence', marker='x', linestyle='None', markersize=3, alpha=0.7)
    plt.xlabel("Question Index (Sorted by Real Image Confidence Descending)")
    plt.ylabel("Top Softmax Probability")
    plt.title("Confidence Drop Comparison: Real vs. Dummy Images (Markers Only)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0, 1.05)
    try:
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"Confidence comparison marker plot saved to: {output_plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()
    return merged_df, average_drop

def plot_probability_margin_histogram(real_df: pd.DataFrame,
                                      dummy_df: pd.DataFrame,
                                      output_plot_path: str = "prob_margin_histogram.png",
                                      num_bins: int = 20):
    if real_df.empty or dummy_df.empty:
        print("Warning: One or both input DataFrames are empty. Skipping probability margin histogram.")
        return
    real_margins = real_df.loc[real_df['prob_margin'].notna(), 'prob_margin']
    dummy_margins = dummy_df.loc[dummy_df['prob_margin'].notna(), 'prob_margin']
    if real_margins.empty and dummy_margins.empty:
        print("Warning: No valid probability margins found in either dataset. Skipping histogram.")
        return
    bin_edges = np.linspace(0, 1, num_bins + 1)
    plt.figure(figsize=(10, 6))
    plt.hist(real_margins, bins=bin_edges, alpha=0.6, label='Real Images', density=True)
    plt.hist(dummy_margins, bins=bin_edges, alpha=0.6, label='Dummy Images', density=True)
    plt.xlabel("Margin (Top Probability - Second Probability)")
    plt.ylabel("Density")
    plt.title("Distribution of Prediction Probability Margins: Real vs. Dummy Images")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, 1)
    try:
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"Probability margin histogram saved to: {output_plot_path}")
    except Exception as e:
        print(f"Error saving probability margin histogram: {e}")
    plt.close()

def plot_accuracy_vs_confidence(df: pd.DataFrame,
                                output_plot_path: str = "accuracy_vs_confidence.png",
                                num_threshold_steps: int = 20,
                                plot_title: str = 'Accuracy vs. Confidence Threshold'):
    if df.empty:
        print(f"Warning: Input DataFrame is empty. Skipping plot: {plot_title}")
        return
    thresholds = np.linspace(0.0, 1.0, num_threshold_steps + 1)
    accuracies = []
    retained_fractions = []
    total_samples = len(df)
    if total_samples == 0:
        print(f"Warning: Zero samples in DataFrame. Skipping plot: {plot_title}")
        return
    print(f"\nCalculating Accuracy vs. Confidence for: {plot_title}")
    for thresh in tqdm(thresholds, desc="Thresholds"):
        filtered_df = df[df['top_confidence'] >= thresh]
        retained_count = len(filtered_df)
        retained_fraction = retained_count / total_samples
        retained_fractions.append(retained_fraction)
        if retained_count == 0:
            accuracies.append(np.nan)
        else:
            accuracy = filtered_df['is_correct_raw'].mean()
            accuracies.append(accuracy)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Accuracy (on retained samples)', color=color)
    ax1.plot(thresholds, accuracies, color=color, marker='o', linestyle='-', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, axis='y', linestyle='--', linewidth=0.5)
    overall_accuracy = df['is_correct_raw'].mean()
    ax1.axhline(overall_accuracy, color='gray', linestyle=':', label=f'Overall Acc ({overall_accuracy:.3f})')
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Fraction of Samples Retained', color=color)
    ax2.plot(thresholds, retained_fractions, color=color, marker='x', linestyle='--', label='Fraction Retained')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.05)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center left')
    plt.title(plot_title)
    plt.xlim(0, 1)
    ax1.grid(True, axis='x', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    try:
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy vs. Confidence curve saved to: {output_plot_path}")
    except Exception as e:
        print(f"Error saving accuracy vs. confidence plot: {e}")
    plt.close()

# --- Main Execution (Modified for Saving & New Metrics) ---
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
    parser.add_argument("--acc-conf-curve-real", type=str, default="acc_vs_conf_real.png", help="Filename for Accuracy vs Confidence curve (Real).")
    parser.add_argument("--acc-conf-curve-dummy", type=str, default="acc_vs_conf_dummy.png", help="Filename for Accuracy vs Confidence curve (Dummy).")

    cli_args = parser.parse_args()
    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving analysis results to: {output_dir.resolve()}")

    real_processed_df = analyze_prefix_scores(cli_args.real_results)
    dummy_processed_df = analyze_prefix_scores(cli_args.dummy_results)

    # Save individual DataFrames
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

    merged_comparison_df, avg_conf_drop = compare_and_plot_confidence(
        real_processed_df, dummy_processed_df, output_dir / cli_args.plot_output
    )
    if not merged_comparison_df.empty:
        compare_csv_path = output_dir / cli_args.compare_csv
        try:
            merged_comparison_df.to_csv(compare_csv_path, index=False)
            print(f"Comparison metrics saved to: {compare_csv_path}")
        except Exception as e:
            print(f"Error saving comparison CSV: {e}")

    plot_probability_margin_histogram(
        real_processed_df, dummy_processed_df, output_dir / cli_args.margin_hist_output
    )
    if not real_processed_df.empty:
        plot_accuracy_vs_confidence(
            real_processed_df, output_plot_path=output_dir / cli_args.acc_conf_curve_real,
            plot_title='Accuracy vs. Confidence Threshold (Real Images)'
        )
    # Plot for dummy if needed
    # if not dummy_processed_df.empty:
    #     plot_accuracy_vs_confidence(
    #         dummy_processed_df, output_plot_path=output_dir / cli_args.acc_conf_curve_dummy,
    #         plot_title='Accuracy vs. Confidence Threshold (Dummy Images)'
    #     )

    summary_data = {
        "input_files": { "real": cli_args.real_results, "dummy": cli_args.dummy_results },
        "threshold_scheme": {
            "confidence_thresholds_adaptive": ADAPTIVE_CONF_THRESHOLDS, # Clarified name
            "raw_score_margin_threshold_for_acc": DEFAULT_MARGIN_THRESHOLD 
        },
        "real_image_summary": {}, "dummy_image_summary": {}, "comparison_summary": {}
    }

    # --- Calculate and add new margin metrics to summary ---
    for df, df_name_key in [(real_processed_df, "real_image_summary"), (dummy_processed_df, "dummy_image_summary")]:
        if not df.empty:
            # Calculate global average probability margin
            # Ensure 'prob_margin' column exists and has non-NaN values
            valid_prob_margins = df.loc[df['prob_margin'].notna(), 'prob_margin']
            global_avg_prob_margin = valid_prob_margins.mean() if not valid_prob_margins.empty else np.nan

            # Calculate average probability margin for correct answers
            correct_items_with_margin = df.loc[df['is_correct_raw'] & df['prob_margin'].notna(), 'prob_margin']
            avg_prob_margin_for_correct = correct_items_with_margin.mean() if not correct_items_with_margin.empty else np.nan

            # Calculate average probability margin for INCORRECT answers 
            incorrect_items_with_margin = df.loc[(~df['is_correct_raw']) & df['prob_margin'].notna(), 'prob_margin'] # Note the ~ for NOT correct
            avg_prob_margin_for_incorrect = incorrect_items_with_margin.mean() if not incorrect_items_with_margin.empty else np.nan

            # Average top confidence (softmax probability) for CORRECT answers 
            correct_confidences = df.loc[df['is_correct_raw'] & df['top_confidence'].notna(), 'top_confidence']
            avg_top_confidence_correct = correct_confidences.mean() if not correct_confidences.empty else np.nan

            # Average top confidence (softmax probability) for INCORRECT answers 
            incorrect_confidences = df.loc[(~df['is_correct_raw']) & df['top_confidence'].notna(), 'top_confidence']
            avg_top_confidence_incorrect = incorrect_confidences.mean() if not incorrect_confidences.empty else np.nan
            
            summary_data[df_name_key] = {
                "raw_accuracy": df['is_correct_raw'].mean(),
                "adaptive_confidence_accuracy": df['is_correct_adaptive_confidence'].mean(),
                "adaptive_margin_accuracy_from_raw_scores": df['is_correct_adaptive_margin'].mean(), # Clarified name
                "average_top_confidence_softmax_prob": df['top_confidence'].mean(), # Clarified name
                "average_entropy": df['entropy'].mean(),
                "global_average_prob_margin": global_avg_prob_margin, 
                "average_prob_margin_for_correct_answers": avg_prob_margin_for_correct,
                "average_prob_margin_for_incorrect_answers": avg_prob_margin_for_incorrect,
                "average_top_confidence_for_correct_answers": avg_top_confidence_correct,   
                "average_top_confidence_for_incorrect_answers": avg_top_confidence_incorrect,
                "count": len(df)
            }
            print(f"\n--- {df_name_key.replace('_', ' ').title()} Summary ---")
            for key, value in summary_data[df_name_key].items():
                 print(f"{key}: {value:.4f}" if isinstance(value, float) and not np.isnan(value) else f"{key}: {value}")
            print("--------------------------------------------------------")
        else:
            summary_data[df_name_key] = { "count": 0 } 

    summary_data["comparison_summary"] = {
            "average_confidence_drop": avg_conf_drop if not np.isnan(avg_conf_drop) else None,
            "compared_question_count": len(merged_comparison_df) if not merged_comparison_df.empty else 0
    }
    if not merged_comparison_df.empty : # Check if comparison was possible
        print("\n--- Comparison Summary ---")
        print(f"Average Confidence Drop (Real - Dummy): {avg_conf_drop:.4f}" if not np.isnan(avg_conf_drop) else "N/A")
        print(f"Number of Questions Compared: {len(merged_comparison_df)}")
        print("--------------------------")

    summary_json_path = output_dir / cli_args.summary_json
    try:
        with open(summary_json_path, 'w') as f:
            json.dump(summary_data, f, indent=4, cls=NpEncoder) 
        print(f"Summary metrics saved to: {summary_json_path}")
    except Exception as e:
        print(f"Error saving summary JSON: {e}")


# /home/alex.ia/miniconda3/envs/blip2_lavis_env/bin/python /home/alex.ia/OmniMed/scripts/fine-tuning/analyse_finetune_scores.py \
#     --real-results "/home/alex.ia/OmniMed/results_eval_ft/blip2-flant5xl-ft-fundus-ep5_real_prefix/x-ray/x-ray_prefix_results.json" \
#     --dummy-results "/home/alex.ia/OmniMed/results_eval_ft/dummy/blip2-flant5xl-ft-fundus-ep5_dummy_prefix/x-ray/x-ray_prefix_results.json" \
#     --output-dir "/home/alex.ia/OmniMed/results_eval_ft/analysis_metrics/x-ray_ep5" \
#     --real-csv "real_metrics_x-ray_ft_ep5.csv" \
#     --dummy-csv "dummy_metrics_x-ray_ft_ep5.csv" \
#     --compare-csv "comparison_metrics_x-ray_ft_ep5.csv" \
#     --summary-json "summary_metrics_x-ray_ft_ep5.json" \
#     --plot-output "confidence_drop_plot_x-ray_ft_ep5.png" \
#     --margin-hist-output "margin_histogram_x-ray_ft_ep5.png" \
#     --acc-conf-curve-real "acc_vs_conf_real_x-ray_ft_ep5.png" \
#     --acc-conf-curve-dummy "acc_vs_conf_dummy_x-ray_ft_ep5.png"