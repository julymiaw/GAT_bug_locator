import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Optional

# Assuming ExperimentTracker is in the same directory or accessible via PYTHONPATH
try:
    # If experiment_tracker is part of a package, adjust the import accordingly
    # e.g., from your_package import ExperimentTracker
    from experiment_tracker import ExperimentTracker
except ImportError:
    print(
        "Error: experiment_tracker.py not found. Make sure it's in the same directory or your PYTHONPATH."
    )
    exit(1)

# --- Configuration ---
# Define ablation categories and their plotting order
ABLATION_CATEGORIES = {
    "MLP (No Graph)": 1,  # Baseline: heads=None
    "GAT (Self-Loops Only)": 2,  # Ablation: heads!=None, use_self_loops_only=True
    "GAT (Full Edges)": 3,  # Full Model: heads!=None, use_self_loops_only=False
}
CATEGORY_ORDER = ["MLP (No Graph)", "GAT (Self-Loops Only)", "GAT (Full Edges)"]
# --- End Configuration ---


def determine_ablation_category(row) -> Optional[str]:
    """Determines the ablation category based on model parameters."""
    heads = row.get("heads")
    use_self_loops = row.get(
        "use_self_loops_only", False
    )  # Default to False if missing

    # Handle potential string 'None' or actual None/NaN
    is_mlp = pd.isna(heads) or str(heads).lower() == "none"

    if is_mlp:
        return "MLP (No Graph)"
    elif use_self_loops:
        return "GAT (Self-Loops Only)"
    else:  # heads is not None/NaN/'None' and use_self_loops is False
        return "GAT (Full Edges)"


def load_and_process_all_datasets(
    dataset_prefixes: List[str],
) -> Optional[pd.DataFrame]:
    """Loads data from multiple datasets and processes it for ablation study."""
    all_datasets_data = []
    print("Starting data loading process...")

    for prefix in dataset_prefixes:
        print(f"\n--- Processing Dataset: {prefix} ---")
        dataset_data = []
        # Load results from both MAP and MRR optimized runs
        for metric_type in ["MAP", "MRR"]:
            log_dir = f"{prefix}_GAT_{metric_type}"
            log_file = os.path.join(log_dir, "Adaptive_models_registry.json")

            if not os.path.exists(log_file):
                print(
                    f"Warning: Log file not found for {prefix} ({metric_type}): {log_file}"
                )
                continue

            print(f"Loading logs from: {log_dir}")
            try:
                tracker = ExperimentTracker()
                tracker.from_json(log_file)
                df = tracker.to_dataframe()
                if df.empty:
                    print(f"Warning: No model data found in log file: {log_file}")
                    continue
                df["optimization_metric"] = metric_type  # Track which run it came from
                df["dataset"] = prefix  # Add dataset identifier
                dataset_data.append(df)
                print(f"✓ Loaded {len(df)} model results for {prefix} ({metric_type}).")
            except Exception as e:
                print(f"✗ Error loading or processing logs from {log_dir}: {e}")
                continue

        if not dataset_data:
            print(f"Warning: No log data could be loaded for dataset {prefix}.")
            continue

        # Combine MAP and MRR optimized runs for this dataset
        combined_df_dataset = pd.concat(dataset_data, ignore_index=True)
        all_datasets_data.append(combined_df_dataset)

    if not all_datasets_data:
        print("Error: No log data could be loaded for ANY specified dataset.")
        return None

    # Combine data from all datasets
    final_combined_df = pd.concat(all_datasets_data, ignore_index=True)
    print(
        f"\nTotal combined data entries across all datasets: {len(final_combined_df)}"
    )

    # Determine ablation category for each entry
    final_combined_df["ablation_category"] = final_combined_df.apply(
        determine_ablation_category, axis=1
    )

    # Filter out entries that don't belong to a defined ablation category
    valid_categories = list(ABLATION_CATEGORIES.keys())
    filtered_df = final_combined_df[
        final_combined_df["ablation_category"].isin(valid_categories)
    ].copy()

    if filtered_df.empty:
        print(
            "Error: No models matching the defined ablation categories found in the logs across all datasets."
        )
        print(
            "Please check the 'heads' and 'use_self_loops_only' parameters in your logs."
        )
        return None

    print(f"Filtered down to {len(filtered_df)} relevant ablation model entries.")
    # print("\nSample of filtered data with ablation categories:")
    # print(filtered_df[['dataset', 'model_id', 'heads', 'use_self_loops_only', 'ablation_category']].head())
    return filtered_df


def calculate_average_scores(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Calculates the average score for each dataset and ablation category.
    Strategy: Finds the best score per fold for each category/dataset, then averages these best scores.
    """
    score_col = f"predict_{metric}_score"
    if score_col not in df.columns:
        print(f"Error: Score column '{score_col}' not found in DataFrame.")
        return pd.DataFrame()

    # Ensure score column is numeric, coercing errors
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    # Drop rows where the score is NaN after coercion or essential columns are missing
    essential_cols = ["dataset", "fold_num", "ablation_category", score_col]
    df = df.dropna(subset=essential_cols)

    if df.empty:
        print(
            f"Warning: No valid data remaining after handling missing values for metric {metric}."
        )
        return pd.DataFrame()

    # --- Strategy: Find best score per fold first, then average ---
    # 1. Find the best score for each fold, category, and dataset combination.
    #    This handles cases where multiple runs (e.g., different hyperparameters within the same category) exist for the same fold.
    #    We use idxmax() to get the index of the row with the maximum score within each group.
    try:
        idx_best_per_fold = df.loc[
            df.groupby(["dataset", "fold_num", "ablation_category"])[score_col].idxmax()
        ]
    except KeyError as e:
        print(
            f"KeyError during groupby/idxmax: {e}. This might happen if a group is empty after filtering."
        )
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during groupby/idxmax: {e}")
        return pd.DataFrame()

    if idx_best_per_fold.empty:
        print(
            f"Warning: No best scores could be determined per fold for metric {metric}."
        )
        return pd.DataFrame()

    # 2. Calculate the average of these best scores across folds for each dataset and category
    average_scores = (
        idx_best_per_fold.groupby(["dataset", "ablation_category"], observed=True)[
            score_col
        ]
        .mean()
        .reset_index()
    )

    print(f"\nCalculated average scores for metric: {metric}")
    print(average_scores)

    return average_scores


def plot_multi_dataset_ablation(
    dataset_prefixes: List[str], avg_scores_df: pd.DataFrame, metric: str
):
    """Generates and saves the multi-dataset ablation comparison bar chart (in English)."""
    score_col = f"predict_{metric}_score"

    # Check if the score column exists in the averaged data
    if score_col not in avg_scores_df.columns:
        print(
            f"Error: Column '{score_col}' not found in the averaged scores DataFrame. Cannot plot."
        )
        return

    # Pivot the data for plotting
    try:
        plot_data = avg_scores_df.pivot(
            index="dataset", columns="ablation_category", values=score_col
        )
    except Exception as e:
        print(f"Error pivoting data for plotting: {e}")
        return

    # Reorder columns based on CATEGORY_ORDER and rows based on input dataset_prefixes
    # Use .reindex() to handle potentially missing categories or datasets gracefully (filling with 0 or NaN)
    plot_data = plot_data.reindex(columns=CATEGORY_ORDER, fill_value=0)
    plot_data = plot_data.reindex(
        index=dataset_prefixes, fill_value=0
    )  # Ensure dataset order matches input

    if plot_data.empty:
        print(
            f"Warning: No data available to plot for metric {metric} after pivoting and reindexing."
        )
        return

    n_datasets = len(plot_data.index)
    n_categories = len(plot_data.columns)

    if n_datasets == 0 or n_categories == 0:
        print(
            f"Warning: Zero datasets or categories found in the data to plot for metric {metric}."
        )
        return

    bar_width = 0.8 / n_categories  # Adjust bar width based on number of categories
    index = np.arange(n_datasets)

    fig, ax = plt.subplots(
        figsize=(max(10, n_datasets * 1.5), 6)
    )  # Adjust width based on number of datasets

    # Define colors for categories
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_categories))  # Use viridis colormap
    category_colors = {cat: colors[i] for i, cat in enumerate(plot_data.columns)}

    # Plot bars for each category
    for i, category in enumerate(plot_data.columns):
        scores = plot_data[category]
        bars = ax.bar(
            index + i * bar_width,
            scores,
            bar_width,
            label=category,
            color=category_colors[category],
        )
        # Add value labels on top of bars
        ax.bar_label(bars, fmt="%.3f", label_type="edge", fontsize=8, padding=3)

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel(f"Average {metric} Score", fontsize=12)
    ax.set_title(f"Ablation Study Comparison ({metric}) Across Datasets", fontsize=14)
    ax.set_xticks(index + bar_width * (n_categories - 1) / 2)
    ax.set_xticklabels(
        plot_data.index, rotation=45, ha="right"
    )  # Rotate labels if many datasets
    ax.legend(title="Model Variant")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.3f}")
    )  # Format y-axis labels

    # Adjust y-limit to make space for labels
    current_ylim = ax.get_ylim()
    ax.set_ylim(
        bottom=0, top=current_ylim[1] * 1.15
    )  # Increase top margin slightly more

    plt.tight_layout()

    # Save the plot
    output_filename = f"ablation_comparison_{metric}.png"
    try:
        plt.savefig(output_filename)
        print(f"✓ Saved multi-dataset ablation plot: {output_filename}")
    except Exception as e:
        print(f"✗ Error saving plot {output_filename}: {e}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Multi-Dataset Ablation Study Plots (Section 6.4) in English."
    )
    parser.add_argument(
        "dataset_prefixes",
        nargs="+",
        help="List of dataset prefixes (e.g., 'AspectJ' 'Tomcat' 'SWT' 'BIRT'). "
        "Assumes log directories are named '{prefix}_GAT_MAP' and '{prefix}_GAT_MRR'.",
    )
    args = parser.parse_args()

    # Load and process data from all specified datasets
    processed_data = load_and_process_all_datasets(args.dataset_prefixes)

    if processed_data is None or processed_data.empty:
        print("Exiting due to data loading or processing errors.")
        return

    # Calculate average scores for MAP and MRR
    avg_map_scores = calculate_average_scores(processed_data.copy(), "MAP")
    avg_mrr_scores = calculate_average_scores(processed_data.copy(), "MRR")

    # Generate plots using the calculated average scores
    if not avg_map_scores.empty:
        plot_multi_dataset_ablation(args.dataset_prefixes, avg_map_scores, "MAP")
    else:
        print("Skipping MAP plot due to empty average scores.")

    if not avg_mrr_scores.empty:
        plot_multi_dataset_ablation(args.dataset_prefixes, avg_mrr_scores, "MRR")
    else:
        print("Skipping MRR plot due to empty average scores.")


if __name__ == "__main__":
    main()
