#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import logging
import sys
import matplotlib.font_manager as fm

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add Times New Roman font
font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
times_new_roman = fm.FontProperties(fname=font_path)

# Set up matplotlib to use Times New Roman font
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
    }
)


def load_results(filepath):
    """Load results from CSV file."""
    filepath = Path(filepath)
    logger.info(f"Loading results from {filepath}")

    # Check if file exists
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        logger.error("Please check the file path and try again")
        sys.exit(1)

    # Check file extension
    if filepath.suffix.lower() not in [".csv"]:
        logger.warning(f"File {filepath} does not have a .csv extension")
        logger.warning("Make sure this is the correct file")

    try:
        results = pd.read_csv(filepath)

        # Verify required columns exist
        required_columns = ["method", "alpha", "empirical_factuality", "removal_fraction"]
        missing_columns = [col for col in required_columns if col not in results.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.error(
                "The CSV file must contain: method, alpha, empirical_factuality, removal_fraction"
            )
            sys.exit(1)

        return results
    except pd.errors.EmptyDataError:
        logger.error(f"The file {filepath} is empty")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        sys.exit(1)


def get_method_label(method):
    """Convert method name to display label."""
    method_mapping = {"textual-bayes": "MHLP (ours) frequency scoring", "gpt-4": "GPT-4 frequency scoring"}
    return method_mapping.get(method, method)


# Define color scheme
# COLOR_SCHEME = {
#     'blue': '#2E86C1',      # Professional blue
#     'orange': '#E67E22',    # Warm orange
#     'green': '#27AE60',     # Forest green
#     'purple': '#8E44AD',    # Deep purple
#     'red': '#C0392B'
# }

COLOR_SCHEME = {
    "blue": "#567895",
    "pink": "#d43939",
    "yellow": "#ffc314",
    "green": "#71b05b",
    "purple": "#745baf",
}


def plot_factuality(results, save_dir=None, exp_name="experiment", calib_size=25):
    """Plot empirical factuality results."""
    methods = results["method"].unique()

    plt.figure(figsize=(6, 4))

    # First get all unique x values from the data
    all_x_values = set()
    for method in methods:
        method_results = results[results["method"] == method]
        grouped_results = (
            method_results.groupby("alpha")["empirical_factuality"].agg(["mean"]).reset_index()
        )
        x = 1 - grouped_results["alpha"]
        all_x_values.update(x)

    # Convert to sorted list for consistent plotting
    x_values = sorted(list(all_x_values))

    # Plot each method's results
    for i, method in enumerate(methods):
        method_results = results[results["method"] == method]
        grouped_results = (
            method_results.groupby("alpha")["empirical_factuality"].agg(["mean"]).reset_index()
        )
        x = 1 - grouped_results["alpha"]
        y = grouped_results["mean"]

        color = list(COLOR_SCHEME.values())[i % len(COLOR_SCHEME)]
        # Use dotted line for MHLP (textual-bayes)
        linestyle = ":" if method == "textual-bayes" else "-"
        plt.plot(
            x,
            y,
            f"o{linestyle}",
            label=get_method_label(method),
            markersize=5,
            linewidth=2,
            color=color,
        )

    # Add ideal and upper bound lines using exact x points
    y_values = np.array(x_values) + 1 / (calib_size + 1)
    plt.plot(x_values, x_values, "--", color="gray", label="Conformal bounds", linewidth=2)
    plt.plot(x_values, y_values, "--", color="gray", linewidth=2)

    plt.xlabel("Target Factuality (1-α)", fontsize=14, fontproperties=times_new_roman)
    plt.ylabel("Empirical Factuality", fontsize=14, fontproperties=times_new_roman)
    plt.legend(fontsize=12, prop=times_new_roman)

    # Set tick labels to use Times New Roman
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(times_new_roman)

    if save_dir:
        plt.savefig(save_dir / f"conformal_factuality_coverage.pdf", bbox_inches="tight")
        plt.savefig(save_dir / f"conformal_factuality_coverage.png", bbox_inches="tight")

    plt.close()


def plot_removal_rate(results, save_dir=None, exp_name="experiment"):
    """Plot removal rate results with error bars using standard error."""
    methods = results["method"].unique()

    plt.figure(figsize=(6, 4))
    for i, method in enumerate(methods):
        method_results = results[results["method"] == method]

        # Group by alpha and calculate mean and standard error
        grouped = method_results.groupby("alpha")
        x_values = []
        y_values = []
        yerr_values = []

        for alpha, group in grouped:
            x_values.append(group["empirical_factuality"].mean())
            y_values.append(group["removal_fraction"].mean())
            yerr_values.append(np.std(group["removal_fraction"]))

        # Sort by x values for proper line plotting
        sorted_indices = np.argsort(x_values)
        x_values = np.array(x_values)[sorted_indices]
        y_values = np.array(y_values)[sorted_indices]
        yerr_values = np.array(yerr_values)[sorted_indices]

        color = list(COLOR_SCHEME.values())[i % len(COLOR_SCHEME)]
        # Use dotted line for MHLP (textual-bayes)
        linestyle = ":" if method == "textual-bayes" else "-"
        # Plot with error bars
        plt.errorbar(
            x_values,
            y_values,
            yerr=yerr_values,
            fmt=f"o{linestyle}",
            capsize=5,
            label=get_method_label(method),
            markersize=5,
            linewidth=2,
            capthick=2,
            color=color,
            ecolor=color,
        )

    plt.xlabel("Empirical Factuality", fontsize=14, fontproperties=times_new_roman)
    plt.ylabel("Average Percent Removed", fontsize=14, fontproperties=times_new_roman)
    plt.legend(fontsize=12, prop=times_new_roman)

    # Set tick labels to use Times New Roman
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(times_new_roman)

    if save_dir:
        plt.savefig(save_dir / f"conformal_factuality_removal.pdf", bbox_inches="tight")
        plt.savefig(save_dir / f"conformal_factuality_removal.png", bbox_inches="tight")

    plt.close()


def plot_removal_rate_with_std(results, save_dir=None, exp_name="experiment"):
    """Plot removal rate results with standard error bars calculated from individual removal fractions."""
    methods = results["method"].unique()

    plt.figure(figsize=(6, 4))
    for i, method in enumerate(methods):
        method_results = results[results["method"] == method]

        # Group by alpha and calculate mean and standard error
        grouped = method_results.groupby("alpha")
        x_values = []
        y_values = []
        yerr_values = []

        for alpha, group in grouped:
            x_values.append(group["empirical_factuality"].mean())
            # Calculate mean and std from individual removal fractions
            all_removal_fractions = []
            for fractions in group["removal_fractions"]:
                # Convert string fractions to float if needed
                if isinstance(fractions, str):
                    try:
                        fractions = eval(
                            fractions
                        )  # Convert string representation of list to actual list
                    except:
                        fractions = [float(f) for f in fractions.strip("[]").split(",")]
                all_removal_fractions.extend(fractions)
            y_values.append(np.mean(all_removal_fractions))
            yerr_values.append(
                1.96 * np.std(all_removal_fractions) / np.sqrt(len(all_removal_fractions))
            )

        # Sort by x values for proper line plotting
        sorted_indices = np.argsort(x_values)
        x_values = np.array(x_values)[sorted_indices]
        y_values = np.array(y_values)[sorted_indices]
        yerr_values = np.array(yerr_values)[sorted_indices]

        color = list(COLOR_SCHEME.values())[i % len(COLOR_SCHEME)]
        # Use dotted line for MHLP (textual-bayes)
        linestyle = ":" if method == "textual-bayes" else "-"
        # Plot with error bars
        plt.errorbar(
            x_values,
            y_values,
            yerr=yerr_values,
            fmt=f"o{linestyle}",
            capsize=5,
            label=get_method_label(method),
            markersize=5,
            linewidth=2,
            capthick=2,
            color=color,
            ecolor=color,
        )

    plt.xlabel("Empirical Factuality", fontsize=14, fontproperties=times_new_roman)
    plt.ylabel("Average Percent Removed", fontsize=14, fontproperties=times_new_roman)
    plt.legend(fontsize=12, prop=times_new_roman)

    # Set tick labels to use Times New Roman
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(times_new_roman)

    if save_dir:
        plt.savefig(
            save_dir / f"conformal_factuality_removal_std_{exp_name}.pdf", bbox_inches="tight"
        )
        plt.savefig(
            save_dir / f"conformal_factuality_removal_std_{exp_name}.png", bbox_inches="tight"
        )

    plt.close()


def plot_conformal_results(results, save_dir=None, exp_name="experiment", calib_size=25):
    """Plot all conformal factuality results."""
    methods = results["method"].unique()
    logger.info(f"Plotting results for methods: {methods}")

    # Create save directory if specified
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving plots to {save_dir}")

    # Plot all types of results
    plot_factuality(results, save_dir, exp_name, calib_size)
    plot_removal_rate(results, save_dir, exp_name)
    plot_removal_rate_with_std(results, save_dir, exp_name)


def print_statistics(results):
    """Print basic statistics about the results."""
    print("\n=== Results Statistics ===")
    print(f"Available methods: {results['method'].unique()}")
    print(f"\nAlpha values: {sorted(results['alpha'].unique())}")
    print("\nSummary statistics:")
    print(results.groupby("method")[["empirical_factuality", "removal_fraction"]].describe())


def main():
    parser = argparse.ArgumentParser(description="Plot conformal factuality results from CSV file.")
    parser.add_argument("results_path", type=str, help="Path to the CSV results file")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/plots",
        help="Directory to save the plots (default: outputs/plots)",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="experiment",
        help="Experiment name to use in saved plot filenames",
    )
    parser.add_argument("--no-stats", action="store_true", help="Skip printing statistics")
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["all", "factuality", "removal", "removal_std"],
        default="all",
        help="Which type of plot to generate",
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        default=25,
        help="Calibration set size for upper bound calculation (default: 25)",
    )

    args = parser.parse_args()

    # Load and plot results
    try:
        results = load_results(args.results_path)

        if not args.no_stats:
            print_statistics(results)

        save_dir = Path(args.save_dir) if args.save_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving plots to {save_dir}")

        if args.plot_type == "all":
            plot_conformal_results(results, save_dir, args.exp_name, args.calib_size)
        elif args.plot_type == "factuality":
            plot_factuality(results, save_dir, args.exp_name, args.calib_size)
        elif args.plot_type == "removal":
            plot_removal_rate(results, save_dir, args.exp_name)
        elif args.plot_type == "removal_std":
            plot_removal_rate_with_std(results, save_dir, args.exp_name)

        logger.info("Plotting completed successfully")

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
