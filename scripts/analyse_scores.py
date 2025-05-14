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
import argparse
from pathlib import Path

# --- Define Adaptive Threshold Mappings ---
ADAPTIVE_CONF_THRESHOLDS = {
    2: 0.6,
    3: 0.5,
    4: 0.4,
    'default': 0.35
}
DEFAULT_MARGIN_THRESHOLD = 0.15

# --- Helper functions (parse_score_string, get_options_and_scores, calculate_softmax_probs) ---
# These functions remain unchanged from your provided script.
def parse_score_string(score_str: str) -> List[float]:
    try:
        scores = ast.literal_eval(score_str)
        if isinstance(scores, list) and all(isinstance(x, (int, float)) for x in scores):
            return [float(s) for s in scores]
        return []
    except (ValueError, SyntaxError, TypeError):
        return []

def get_options_and_scores(item: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    options = []
    scores_str = item.get("confidence", "[]") # Assumes raw NLL scores are in "confidence"
    scores = parse_score_string(scores_str)
    for key in ["option_A", "option_B", "option_C", "option_D"]:
        option_text = item.get(key)
        if option_text is not None:
            options.append(str(option_text)) # Ensure options are strings
    if len(options) != len(scores):
        return [], []
    return options, scores

def calculate_softmax_probs(scores: List[float]) -> np.ndarray:
    if not scores: return np.array([])
    neg_scores = -np.array(scores)
    probabilities = scipy_softmax(neg_scores)
    return probabilities

# calculate_metrics_for_item remains unchanged as it already calculates
# 'is_correct_raw', 'top_confidence', and 'prob_margin' which we need.
def calculate_metrics_for_item(item: Dict[str, Any]) -> Dict[str, Any]:
    metrics = {
            "question_id": item.get("question_id", "N/A"),
            "gt_answer": str(item.get("gt_answer")),
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
    metrics["raw_scores"] = [float(rs) for rs in raw_scores] # Ensure float
    
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

        effective_margin_thresh = DEFAULT_MARGIN_THRESHOLD
        metrics["effective_margin_threshold"] = effective_margin_thresh
        if len(raw_scores) > 1:
            sorted_raw_scores = np.sort(raw_scores)
            score_margin_val = sorted_raw_scores[1] - sorted_raw_scores[0]
            metrics["score_margin"] = float(score_margin_val)
            metrics["is_correct_adaptive_margin"] = bool(is_correct_raw and score_margin_val >= effective_margin_thresh)
        else:
            metrics["score_margin"] = float(np.inf)
            metrics["is_correct_adaptive_margin"] = bool(is_correct_raw and (effective_margin_thresh <= 0 if num_options == 1 else False))

        if num_options > 1:
            sorted_probs = np.sort(softmax_probs_np)[::-1]
            prob_margin_val = float(sorted_probs[0] - sorted_probs[1])
            metrics["prob_margin"] = prob_margin_val
        else:
            metrics["prob_margin"] = np.nan
        
        epsilon = 1e-9
        metrics["entropy"] = float(-np.sum(softmax_probs_np * np.log(softmax_probs_np + epsilon)))
    return metrics

# --- Main Processing Function (analyze_prefix_scores) ---
# This function remains unchanged from your provided script.
def analyze_prefix_scores(results_json_path: str) -> pd.DataFrame:
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

    # Your baseline script uses "pred_dict"
    pred_list = results_data.get("pred_dict", []) 
    if not pred_list:
        print(f"Warning: 'pred_dict' not found or empty in {results_json_path}.")
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
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray): return obj.tolist()
        if pd.isna(obj): return None
        return super(NpEncoder, self).default(obj)

# --- Plotting functions ---
# compare_and_plot_confidence, plot_probability_margin_histogram, plot_accuracy_vs_confidence
# These functions remain unchanged from your provided script. I'll include them for completeness.
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

    # Save individual DataFrames (no changes here)
    if not real_processed_df.empty:
        real_csv_path = output_dir / cli_args.real_csv
        try: real_processed_df.to_csv(real_csv_path, index=False); print(f"Real image metrics saved to: {real_csv_path}")
        except Exception as e: print(f"Error saving real metrics CSV: {e}")
    if not dummy_processed_df.empty:
        dummy_csv_path = output_dir / cli_args.dummy_csv
        try: dummy_processed_df.to_csv(dummy_csv_path, index=False); print(f"Dummy image metrics saved to: {dummy_csv_path}")
        except Exception as e: print(f"Error saving dummy metrics CSV: {e}")

    # Compare, Plot, and get Comparison Results (no changes here)
    merged_comparison_df, avg_conf_drop = compare_and_plot_confidence(
        real_processed_df, dummy_processed_df, output_dir / cli_args.plot_output
    )
    if not merged_comparison_df.empty:
        compare_csv_path = output_dir / cli_args.compare_csv
        try: merged_comparison_df.to_csv(compare_csv_path, index=False); print(f"Comparison metrics saved to: {compare_csv_path}")
        except Exception as e: print(f"Error saving comparison CSV: {e}")

    plot_probability_margin_histogram(
        real_processed_df, dummy_processed_df, output_dir / cli_args.margin_hist_output
    )
    if not real_processed_df.empty:
        plot_accuracy_vs_confidence(
            real_processed_df, output_plot_path=output_dir / cli_args.acc_conf_curve_real,
            plot_title='Accuracy vs. Confidence Threshold (Real Images)'
        )

    summary_data = {
        "input_files": { "real": cli_args.real_results, "dummy": cli_args.dummy_results },
        "threshold_scheme": {
            "confidence_thresholds_adaptive": ADAPTIVE_CONF_THRESHOLDS,
            "raw_score_margin_threshold_for_acc": DEFAULT_MARGIN_THRESHOLD 
        },
        "real_image_summary": {}, "dummy_image_summary": {}, "comparison_summary": {}
    }

    # --- Calculate and add new metrics to summary ---
    for df, df_name_key in [(real_processed_df, "real_image_summary"), (dummy_processed_df, "dummy_image_summary")]:
        if not df.empty:
            # Average probability margin for CORRECT answers
            correct_margins = df.loc[df['is_correct_raw'] & df['prob_margin'].notna(), 'prob_margin']
            avg_prob_margin_correct = correct_margins.mean() if not correct_margins.empty else np.nan

            # Average probability margin for INCORRECT answers
            incorrect_margins = df.loc[(~df['is_correct_raw']) & df['prob_margin'].notna(), 'prob_margin']
            avg_prob_margin_incorrect = incorrect_margins.mean() if not incorrect_margins.empty else np.nan

            # Average top confidence for CORRECT answers
            correct_confidences = df.loc[df['is_correct_raw'] & df['top_confidence'].notna(), 'top_confidence']
            avg_top_confidence_correct = correct_confidences.mean() if not correct_confidences.empty else np.nan

            # Average top confidence for INCORRECT answers
            incorrect_confidences = df.loc[(~df['is_correct_raw']) & df['top_confidence'].notna(), 'top_confidence']
            avg_top_confidence_incorrect = incorrect_confidences.mean() if not incorrect_confidences.empty else np.nan
            
            summary_data[df_name_key] = {
                "raw_accuracy": df['is_correct_raw'].mean(),
                "adaptive_confidence_accuracy": df['is_correct_adaptive_confidence'].mean(),
                "adaptive_margin_accuracy_from_raw_scores": df['is_correct_adaptive_margin'].mean(),
                "average_top_confidence_overall": df['top_confidence'].mean(), # Renamed for clarity
                "average_entropy": df['entropy'].mean(),
                "average_prob_margin_for_correct_answers": avg_prob_margin_correct,       # NEW
                "average_prob_margin_for_incorrect_answers": avg_prob_margin_incorrect,   # NEW
                "average_top_confidence_for_correct_answers": avg_top_confidence_correct, # NEW
                "average_top_confidence_for_incorrect_answers": avg_top_confidence_incorrect, # NEW
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
    if not merged_comparison_df.empty : 
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


# /home/alex.ia/miniconda3/envs/blip2_lavis_env/bin/python /home/alex.ia/OmniMed/scripts/analyse_scores.py \
#     --real-results "results/blip2_prefix_test_split/blip2-flant5xl_real_prefix/microscopy_images/blip2/_tmp_omni_microscopy_images_dykj9qau_microscopy_images.json" \
#     --dummy-results "/home/alex.ia/OmniMed/results/blip2_prefix_dummy_test_split/blip2-flant5xl_dummy_prefix/microscopy_images/blip2/_tmp_omni_microscopy_images_uxft9f50_microscopy_images.json" \
#     --output-dir "/home/alex.ia/OmniMed/results/analysis_metrics/microscopy_test_split" \
#     --real-csv "real_metrics_microscopy.csv" \
#     --dummy-csv "dummy_metrics_microscopy.csv" \
#     --compare-csv "comparison_metrics_microscopy.csv" \
#     --summary-json "summary_metrics_microscopy.json" \
#     --plot-output "confidence_drop_plot_microscopy.png" \
#     --margin-hist-output "margin_histogram_microscopy.png" \
#     --acc-conf-curve-real "acc_vs_conf_real_microscopy.png" \
#     --acc-conf-curve-dummy "acc_vs_conf_dummy_microscopy.png"       