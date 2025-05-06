import os
import argparse
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Assuming ExperimentTracker is in the same directory or accessible via PYTHONPATH
try:
    from experiment_tracker import ExperimentTracker
except ImportError:
    print(
        "Error: experiment_tracker.py not found. Make sure it's in the same directory or your PYTHONPATH."
    )
    exit(1)


def load_timing_and_convergence_data(dataset_prefixes: List[str]) -> Dict:
    """
    Loads training time stats and convergence epochs from experiment logs.
    Focuses on GAT/MLP models from the MAP-optimized runs for representative timing.
    """
    all_dataset_stats = {}
    print("Starting training efficiency data loading...")

    for prefix in dataset_prefixes:
        print(f"\n--- Processing Dataset: {prefix} ---")
        # Use MAP-optimized run as representative for timing
        metric_type = "MAP"
        log_dir = f"{prefix}_GAT_{metric_type}"
        log_file = os.path.join(log_dir, "Adaptive_models_registry.json")

        if not os.path.exists(log_file):
            print(
                f"Warning: Log file not found for {prefix} ({metric_type}): {log_file}. Skipping dataset."
            )
            continue

        print(f"Loading logs from: {log_dir}")
        try:
            tracker = ExperimentTracker()
            tracker.from_json(log_file)
            df = tracker.to_dataframe()  # Get model details
            if df.empty:
                print(f"Warning: No model data found in log file: {log_file}")
                continue

            # --- 1. Extract Total Training Time per Fold ---
            fold_times = []
            if hasattr(tracker, "training_time_stats") and tracker.training_time_stats:
                for fold_str, stats in tracker.training_time_stats.items():
                    if "time" in stats:
                        fold_times.append(stats["time"])
                avg_total_time_seconds = np.mean(fold_times) if fold_times else 0
                print(
                    f"✓ Found timing stats. Average total time: {avg_total_time_seconds:.2f}s"
                )
            else:
                print("Warning: No 'training_time_stats' found in log file.")
                avg_total_time_seconds = 0

            # --- 2. Extract Convergence Epochs for GAT/MLP models ---
            # Filter for GAT and MLP models that have convergence info
            nn_models_df = df[
                df["model_type"].isin(["GAT", "MLP"])
                & df[
                    "final_epoch"
                ].notna()  # Check if 'final_epoch' exists and is not NaN
            ].copy()

            if nn_models_df.empty:
                print("Warning: No GAT/MLP models with convergence data found.")
                avg_epochs_to_convergence = 0
            else:
                # Use 'final_epoch' as the indicator of convergence/stopping
                nn_models_df["final_epoch"] = pd.to_numeric(
                    nn_models_df["final_epoch"], errors="coerce"
                )
                valid_epochs = nn_models_df["final_epoch"].dropna()
                avg_epochs_to_convergence = (
                    valid_epochs.mean() if not valid_epochs.empty else 0
                )
                print(
                    f"✓ Found {len(valid_epochs)} converged NN models. Average epochs: {avg_epochs_to_convergence:.1f}"
                )

            # --- 3. Calculate Average Time per Epoch ---
            avg_time_per_epoch = (
                (avg_total_time_seconds / avg_epochs_to_convergence)
                if avg_epochs_to_convergence > 0
                else 0
            )

            # Store results for this dataset
            all_dataset_stats[prefix] = {
                "Avg. Total Time (s)": avg_total_time_seconds,
                "Avg. Epochs to Convergence": avg_epochs_to_convergence,
                "Avg. Time per Epoch (s)": avg_time_per_epoch,
            }

        except Exception as e:
            print(f"✗ Error loading or processing logs from {log_dir}: {e}")
            continue

    if not all_dataset_stats:
        print("Error: No efficiency data could be extracted for ANY specified dataset.")
        return {}

    print("\n✓ Finished extracting efficiency data.")
    return all_dataset_stats


def generate_markdown_table(efficiency_data: Dict) -> str:
    """Generates a Markdown table from the efficiency data."""
    if not efficiency_data:
        return "No efficiency data available to generate table."

    # Create DataFrame
    df = pd.DataFrame.from_dict(efficiency_data, orient="index")
    df.index.name = "Dataset"

    # Format numbers
    df["Avg. Total Time (s)"] = df["Avg. Total Time (s)"].map("{:.1f}".format)
    df["Avg. Epochs to Convergence"] = df["Avg. Epochs to Convergence"].map(
        "{:.1f}".format
    )
    df["Avg. Time per Epoch (s)"] = df["Avg. Time per Epoch (s)"].map("{:.3f}".format)

    # Generate Markdown
    markdown_table = df.to_markdown()

    print("\n--- Training Efficiency Summary (Markdown) ---")
    print(markdown_table)
    return markdown_table


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training efficiency (time, epochs) from experiment logs."
    )
    parser.add_argument(
        "dataset_prefixes",
        nargs="+",
        help="List of dataset prefixes (e.g., 'AspectJ' 'Tomcat' 'SWT' 'BIRT' 'Eclipse' 'JDT'). "
        "Assumes log directories are named '{prefix}_GAT_MAP'.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="training_efficiency_summary.md",
        help="Output file path for the Markdown table (default: training_efficiency_summary.md).",
    )

    args = parser.parse_args()

    # Load and process data
    efficiency_data = load_timing_and_convergence_data(args.dataset_prefixes)

    if efficiency_data:
        # Generate Markdown table
        markdown_table = generate_markdown_table(efficiency_data)

        # Save to file
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(markdown_table)
            print(f"\n✓ Saved Markdown table to: {args.output}")
        except Exception as e:
            print(f"✗ Error saving Markdown table to {args.output}: {e}")


if __name__ == "__main__":
    main()
