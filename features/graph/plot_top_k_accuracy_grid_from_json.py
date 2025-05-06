import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Any

# --- Configuration ---
# Order of datasets for plotting, and their corresponding folder prefixes
# (Ensure these prefixes match your directory names for results)
DATASETS_CONFIG: List[Dict[str, str]] = [
    {"name": "AspectJ", "prefix": "aspectj"},
    {"name": "SWT", "prefix": "swt"},
    {"name": "Tomcat", "prefix": "tomcat"},
    {"name": "BIRT", "prefix": "birt"},
    {
        "name": "Eclipse Platform UI",
        "prefix": "eclipse_platform_ui",
    },  # Adjust if your folder prefix is different (e.g., eclipseui)
    {"name": "JDT", "prefix": "jdt"},
]

# Model configurations: legend name, folder indicator suffix, and plotting style
MODELS_CONFIG: List[Dict[str, Any]] = [
    {
        "name": "Baseline (SGD)",
        "indicator_suffix": "_SGD",
        "style": {"color": "dodgerblue", "marker": "o", "linestyle": "-"},
    },
    {
        "name": "GAT (MAP-Opt)",
        "indicator_suffix": "_GAT_MAP",
        "style": {"color": "forestgreen", "marker": "s", "linestyle": "--"},
    },
    {
        "name": "GAT (MRR-Opt)",
        "indicator_suffix": "_GAT_MRR",
        "style": {"color": "crimson", "marker": "^", "linestyle": ":"},
    },
]

RESULTS_FILENAME = "Adaptive_metrics_results.json"
# --- End Configuration ---


def load_accuracy_scores_from_file(result_file_path: str) -> Optional[List[float]]:
    """Loads Accuracy@k scores (k=1 to 20) from a JSON result file."""
    if not os.path.exists(result_file_path):
        # print(f"Debug: Result file path does not exist: {result_file_path}")
        return None
    try:
        with open(result_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        accuracies_at_k = data.get("accuracy_at_k", {})
        scores = [0.0] * 20  # Initialize 20 scores for Accuracy@1 to Accuracy@20
        for k_val in range(1, 21):
            # JSON keys are strings "1", "2", ...
            scores[k_val - 1] = float(accuracies_at_k.get(str(k_val), 0.0))
        return scores
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {result_file_path}")
        return None
    except Exception as e:
        print(f"Error loading scores from {result_file_path}: {e}")
        return None


def find_and_load_scores(
    dataset_prefix: str, model_indicator_suffix: str
) -> Optional[List[float]]:
    """Finds the result file (handling potential timestamps) and loads scores."""
    base_folder_name = f"{dataset_prefix}{model_indicator_suffix}"

    # 1. Try exact folder name
    exact_result_file = os.path.join(base_folder_name, RESULTS_FILENAME)
    if os.path.exists(exact_result_file):
        # print(f"Found exact match: {exact_result_file}")
        return load_accuracy_scores_from_file(exact_result_file)

    # 2. Try glob pattern for timestamped folders (e.g., aspectj_SGD_20230101)
    glob_pattern = f"{base_folder_name}_*"
    matching_folders = glob.glob(glob_pattern)

    if matching_folders:
        matching_folders.sort(reverse=True)  # Prefer newest if multiple timestamps
        timestamped_result_file = os.path.join(matching_folders[0], RESULTS_FILENAME)
        # print(f"Found glob match: {timestamped_result_file} from folder {matching_folders[0]}")
        return load_accuracy_scores_from_file(timestamped_result_file)

    # print(f"Debug: No result file found for {dataset_prefix} with model indicator {model_indicator_suffix}")
    return None


def plot_top_k_accuracy_grid(output_filename="top_k_accuracy_grid_from_json.png"):
    """
    Plots the Top-k accuracy for multiple datasets and models in a 2x3 grid,
    reading data from JSON files.
    """
    num_datasets = len(DATASETS_CONFIG)
    if num_datasets == 0:
        print("No datasets configured. Exiting.")
        return

    # Adjust subplot layout if fewer than 6 datasets
    ncols = 3
    nrows = (
        num_datasets + ncols - 1
    ) // ncols  # Calculate rows needed, typically 2 for up to 6 datasets

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey=True)
    axes = np.array(axes).flatten()  # Ensure axes is always a flat array

    k_values = np.arange(1, 21)

    for i, dataset_info in enumerate(DATASETS_CONFIG):
        if i >= len(axes):
            break
        ax = axes[i]
        dataset_name = dataset_info["name"]
        dataset_folder_prefix = dataset_info["prefix"]

        ax.set_title(dataset_name, fontsize=14)

        found_any_model_data_for_dataset = False
        for model_info in MODELS_CONFIG:
            model_legend_name = model_info["name"]
            model_folder_suffix = model_info["indicator_suffix"]
            model_plot_style = model_info["style"]

            scores = find_and_load_scores(dataset_folder_prefix, model_folder_suffix)

            if scores:
                ax.plot(
                    k_values,
                    scores,
                    label=model_legend_name,
                    **model_plot_style,
                    markersize=5,
                    linewidth=1.5,
                )
                found_any_model_data_for_dataset = True
            else:
                print(
                    f"Warning: No data loaded for {dataset_name} - {model_legend_name}"
                )

        if not found_any_model_data_for_dataset:
            ax.text(
                0.5,
                0.5,
                "Data not found",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=10,
                color="red",
            )

        ax.set_xlabel("k (Top-k)", fontsize=12)
        if i % ncols == 0:  # Y-label only for the first column of each row
            ax.set_ylabel("Accuracy@k", fontsize=12)

        ax.set_xticks(np.arange(1, 21, 2))  # Ticks at 1, 3, 5, ..., 19
        ax.set_xlim(0.5, 20.5)
        ax.set_ylim(0, 1.05)  # Assuming accuracy is between 0 and 1
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(fontsize=9)  # Adjusted legend font size
        ax.tick_params(axis="both", which="major", labelsize=10)

    # Hide any unused subplots if the number of datasets is not a multiple of ncols
    for j in range(num_datasets, nrows * ncols):
        if j < len(axes):
            fig.delaxes(axes[j])

    fig.suptitle(
        "Top-k Accuracy Comparison Across Datasets (from JSON results)",
        fontsize=18,
        y=0.99 if nrows == 1 else 0.97,
    )  # Adjust y for suptitle
    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95 if nrows == 1 else 0.94]
    )  # Adjust layout to make space for suptitle

    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved as {output_filename}")

    plt.show()  # Display the plot
    plt.close(fig)  # Close the figure window


if __name__ == "__main__":
    # This script assumes it's run from a directory where dataset-specific
    # model result folders (e.g., ./aspectj_SGD/, ./aspectj_GAT_MAP/) are located,
    # or that the paths are correctly pointing to them.

    # Example:
    # Your project root might have:
    # - results/
    #   - aspectj_SGD/Adaptive_metrics_results.json
    #   - aspectj_GAT_MAP/Adaptive_metrics_results.json
    #   - ... (other datasets and models)
    # - scripts/plot_top_k_accuracy_grid_from_json.py
    # If so, you'd run this script from the 'results/' directory, or modify
    # the find_and_load_scores function to prepend "results/" to paths.

    plot_top_k_accuracy_grid()
