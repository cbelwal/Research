# Z-Score Normalization with Pooled Reference
# This class pools data from multiple datasets to create a common reference
# for z-score normalization, enabling meaningful comparisons across datasets.

##########################
# CAUTION: This is not used in the paper, was done for ref. only
##########################
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(topRootPath)
# ----------------------------------------------


class CZScoreWithPooledRef:
    """
    Z-Score normalization using a pooled reference from multiple datasets.

    The pooled reference approach:
    1. Combines all values from multiple datasets into a single pool
    2. Computes the mean and standard deviation of the pooled data
    3. Uses these pooled statistics to normalize each individual dataset
    4. This allows fair comparison of statistics across different algorithms/methods

    Why pooled reference?
    - Different algorithms may produce distances in different ranges
    - Raw means/SDs are not directly comparable across algorithms
    - Z-normalization with pooled reference puts all algorithms on the same scale
    - After normalization, z-mean of 0 means "average" relative to all algorithms
    - Positive z-mean means "above average" (larger distances), negative means "below average"
    """

    def __init__(self):
        # Store the pooled statistics after computation
        self.pooled_mean: Optional[float] = None
        self.pooled_std: Optional[float] = None
        # Store z-normalized values for each dataset (for optional plotting)
        self.z_values_by_dataset: Dict[str, np.ndarray] = {}

    def compute_pooled_statistics(self, datasets: Dict[str, List[float]]) -> Tuple[float, float]:
        """
        Compute pooled mean and standard deviation from multiple datasets.

        Args:
            datasets: Dictionary mapping dataset name/id to list of values
                     e.g., {"alg_2": [0.1, 0.2, ...], "alg_3": [0.15, 0.25, ...]}

        Returns:
            Tuple of (pooled_mean, pooled_std)
        """
        # Combine all values into a single array
        all_values = []
        for dataset_name, values in datasets.items():
            all_values.extend(values)

        all_values = np.array(all_values)

        # Compute pooled statistics
        self.pooled_mean = float(np.mean(all_values))
        self.pooled_std = float(np.std(all_values, ddof=0))  # Population std

        # Handle edge case where std is 0 (all values identical)
        if self.pooled_std == 0:
            self.pooled_std = 1.0  # Avoid division by zero

        return self.pooled_mean, self.pooled_std

    def normalize_datasets(self, datasets: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
        """
        Normalize all datasets using the pooled mean and standard deviation.
        Z-score formula: z = (x - pooled_mean) / pooled_std

        Args:
            datasets: Dictionary mapping dataset name/id to list of values

        Returns:
            Dictionary mapping dataset name/id to numpy array of z-scores
        """
        # Compute pooled statistics if not already done
        if self.pooled_mean is None or self.pooled_std is None:
            self.compute_pooled_statistics(datasets)

        # Normalize each dataset
        self.z_values_by_dataset = {}
        for dataset_name, values in datasets.items():
            values_array = np.array(values)
            z_values = (values_array - self.pooled_mean) / self.pooled_std
            self.z_values_by_dataset[dataset_name] = z_values

        return self.z_values_by_dataset

    def compute_z_normalized_stats(self, datasets: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Compute z-normalized mean and standard deviation for each dataset.

        This is the main function to call for analysis. It:
        1. Pools all data to compute reference statistics
        2. Z-normalizes each dataset using the pooled reference
        3. Computes mean and SD of z-values for each dataset

        Args:
            datasets: Dictionary mapping dataset name/id to list of values
                     e.g., {"alg_2_cosine": [0.1, 0.2, ...], "alg_3_cosine": [0.15, 0.25, ...]}

        Returns:
            Dictionary with structure:
            {
                "dataset_name": {
                    "z_mean": float,    # Mean of z-scores (how far from pooled mean)
                    "z_std": float,     # SD of z-scores (spread relative to pooled SD)
                    "raw_mean": float,  # Original mean for reference
                    "raw_std": float,   # Original SD for reference
                    "n_samples": int    # Number of samples
                },
                ...
            }
        """
        # Normalize all datasets
        z_values_by_dataset = self.normalize_datasets(datasets)

        # Compute statistics for each dataset
        results = {}
        for dataset_name, z_values in z_values_by_dataset.items():
            raw_values = np.array(datasets[dataset_name])
            results[dataset_name] = {
                "z_mean": float(np.mean(z_values)),
                "z_std": float(np.std(z_values, ddof=0)),
                "raw_mean": float(np.mean(raw_values)),
                "raw_std": float(np.std(raw_values, ddof=0)),
                "n_samples": len(z_values)
            }

        return results

    def get_pooled_statistics(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get the computed pooled mean and standard deviation.

        Returns:
            Tuple of (pooled_mean, pooled_std) or (None, None) if not yet computed
        """
        return self.pooled_mean, self.pooled_std

    @staticmethod
    def format_value(value: float) -> str:
        """Format a value to handle very small numbers using scientific notation."""
        if value == 0:
            return "0.000000"
        elif abs(value) < 0.000001:
            return f"{value:.6e}"
        else:
            return f"{value:.6f}"

    def print_z_normalized_stats(self, datasets: Dict[str, List[float]],
                                  title: str = "Z-NORMALIZED STATISTICS") -> Dict[str, Dict[str, float]]:
        """
        Compute and print z-normalized statistics for all datasets.

        Args:
            datasets: Dictionary mapping dataset name/id to list of values
            title: Title to print for the output section

        Returns:
            The computed statistics dictionary
        """
        # Compute statistics
        stats = self.compute_z_normalized_stats(datasets)

        # Print header
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(f"Pooled Mean: {self.format_value(self.pooled_mean)}, "
              f"Pooled SD: {self.format_value(self.pooled_std)}")
        print("-" * 80)

        # Print per-dataset statistics
        for dataset_name, stat in stats.items():
            print(f"\n{dataset_name}:")
            print(f"  Z-Mean: {self.format_value(stat['z_mean']):>12}, "
                  f"Z-SD: {self.format_value(stat['z_std']):>12}")
            print(f"  Raw Mean: {self.format_value(stat['raw_mean']):>10}, "
                  f"Raw SD: {self.format_value(stat['raw_std']):>10}, "
                  f"N: {stat['n_samples']}")

        print("=" * 80)

        return stats

    # ==================== PLOTTING FUNCTIONS ====================

    def plot_kde_all_datasets(self, datasets: Dict[str, List[float]],
                               title: str = "KDE of Z-Normalized Values",
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot Kernel Density Estimation (KDE) curves for z-values of all datasets.
        This visualization helps compare the distribution shapes across algorithms.

        Args:
            datasets: Dictionary mapping dataset name/id to list of values
            title: Title for the plot
            save_path: Optional path to save the figure (if None, displays interactively)
            figsize: Figure size as (width, height) in inches
        """
        # Ensure z-values are computed
        if not self.z_values_by_dataset:
            self.normalize_datasets(datasets)

        # Import scipy for KDE (optional dependency)
        try:
            from scipy import stats as scipy_stats
        except ImportError:
            print("Warning: scipy not available. Using histogram instead of KDE.")
            self._plot_histogram_fallback(title, save_path, figsize)
            return

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Define colors for different datasets
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.z_values_by_dataset)))

        # Plot KDE for each dataset
        for (dataset_name, z_values), color in zip(self.z_values_by_dataset.items(), colors):
            # Compute KDE
            kde = scipy_stats.gaussian_kde(z_values)

            # Create x-axis range based on data
            x_min = min(z_values) - 1
            x_max = max(z_values) + 1
            x_range = np.linspace(x_min, x_max, 500)

            # Plot KDE curve
            ax.plot(x_range, kde(x_range), label=dataset_name, color=color, linewidth=2)

            # Add vertical line for mean
            z_mean = np.mean(z_values)
            ax.axvline(x=z_mean, color=color, linestyle='--', alpha=0.5, linewidth=1)

        # Add reference line at z=0 (pooled mean)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=2, label='Pooled Mean (z=0)')

        # Formatting
        ax.set_xlabel('Z-Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"KDE plot saved to: {save_path}")
        else:
            plt.show()

        plt.close(fig)

    def plot_kde_by_metric(self, cosine_datasets: Dict[str, List[float]],
                            euclidean_datasets: Dict[str, List[float]],
                            title_prefix: str = "KDE of Z-Normalized",
                            save_path_prefix: Optional[str] = None,
                            figsize: Tuple[int, int] = (14, 5)) -> None:
        """
        Plot KDE curves separately for cosine and euclidean distances.
        Creates a side-by-side comparison plot.

        Args:
            cosine_datasets: Dictionary of cosine distance datasets
            euclidean_datasets: Dictionary of euclidean distance datasets
            title_prefix: Prefix for subplot titles
            save_path_prefix: Optional prefix for save path (appends _cosine.png, _euclidean.png)
            figsize: Figure size as (width, height) in inches
        """
        # Import scipy for KDE
        try:
            from scipy import stats as scipy_stats
        except ImportError:
            print("Warning: scipy not available for KDE plotting.")
            return

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Define colors
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(cosine_datasets), len(euclidean_datasets))))

        # Plot Cosine distances
        self._plot_kde_on_axis(ax1, cosine_datasets, colors,
                               f"{title_prefix} Cosine Distances", scipy_stats)

        # Plot Euclidean distances
        self._plot_kde_on_axis(ax2, euclidean_datasets, colors,
                               f"{title_prefix} Euclidean Distances", scipy_stats)

        plt.tight_layout()

        # Save or show
        if save_path_prefix:
            save_path = f"{save_path_prefix}_kde.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"KDE comparison plot saved to: {save_path}")
        else:
            plt.show()

        plt.close(fig)

    def _plot_kde_on_axis(self, ax, datasets: Dict[str, List[float]],
                          colors, title: str, scipy_stats) -> None:
        """
        Helper function to plot KDE on a single axis.

        Args:
            ax: Matplotlib axis object
            datasets: Dictionary of datasets to plot
            colors: Array of colors to use
            title: Title for the subplot
            scipy_stats: scipy.stats module for KDE computation
        """
        # Create a temporary normalizer for this set of datasets
        temp_normalizer = CZScoreWithPooledRef()
        z_values_dict = temp_normalizer.normalize_datasets(datasets)

        for i, (dataset_name, z_values) in enumerate(z_values_dict.items()):
            color = colors[i % len(colors)]

            # Compute KDE
            kde = scipy_stats.gaussian_kde(z_values)

            # Create x-axis range
            x_min = min(z_values) - 1
            x_max = max(z_values) + 1
            x_range = np.linspace(x_min, x_max, 500)

            # Plot
            ax.plot(x_range, kde(x_range), label=dataset_name, color=color, linewidth=2)

            # Add mean line
            z_mean = np.mean(z_values)
            ax.axvline(x=z_mean, color=color, linestyle='--', alpha=0.5, linewidth=1)

        # Reference line at z=0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=2)

        # Formatting
        ax.set_xlabel('Z-Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_histogram_fallback(self, title: str, save_path: Optional[str],
                                  figsize: Tuple[int, int]) -> None:
        """
        Fallback histogram plot when scipy is not available.

        Args:
            title: Title for the plot
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.z_values_by_dataset)))

        for (dataset_name, z_values), color in zip(self.z_values_by_dataset.items(), colors):
            ax.hist(z_values, bins=50, alpha=0.5, label=dataset_name, color=color, density=True)

        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=2, label='Pooled Mean')
        ax.set_xlabel('Z-Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Histogram plot saved to: {save_path}")
        else:
            plt.show()

        plt.close(fig)


# ==================== STANDALONE TEST ====================
if __name__ == "__main__":
    # Test with synthetic data
    print("Testing CZScoreWithPooledRef with synthetic data...")

    # Create synthetic datasets with different distributions
    np.random.seed(42)
    datasets = {
        "Algorithm_2": list(np.random.normal(0.5, 0.1, 1000)),   # Mean 0.5, SD 0.1
        "Algorithm_3": list(np.random.normal(0.6, 0.15, 1000)),  # Mean 0.6, SD 0.15
        "Algorithm_11": list(np.random.normal(0.4, 0.08, 1000)), # Mean 0.4, SD 0.08
        "Algorithm_21": list(np.random.normal(0.55, 0.2, 1000)), # Mean 0.55, SD 0.2
    }

    # Create normalizer and compute statistics
    normalizer = CZScoreWithPooledRef()
    stats = normalizer.print_z_normalized_stats(datasets, "TEST: Z-NORMALIZED STATISTICS")

    # Test KDE plotting
    print("\nGenerating KDE plot...")
    normalizer.plot_kde_all_datasets(datasets, "Test KDE Plot")
