import pandas as pd
import matplotlib.pyplot as plt


class Plots:
    """
    Class for reading SSP transpiled benchmark results and generating comparison plots
    for different assembly types (FullQFT vs HalfQFT).
    """

    def __init__(self, csv_path: str):
        """
        Args:
            csv_path: Path to 'ssp_transpiled_benchmark_results.csv'.
        """
        self.csv_path = csv_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Reads the CSV file into a pandas DataFrame and returns it.

        Returns:
            A DataFrame containing the transpiled benchmark results, with columns:
            ['size', 'instance_id', 'assembly_type', 'num_qubits', 'num_clbits',
             'depth', 'width', 'circuit_size', ...basis gate counts...]
        """
        df = pd.read_csv(self.csv_path)
        # Ensure 'size' and 'assembly_type' are present
        expected_cols = {"size", "assembly_type"}
        if not expected_cols.issubset(set(df.columns)):
            raise ValueError(f"CSV missing required columns: {expected_cols - set(df.columns)}")
        self.df = df
        return df

    def plot_transpiled_stats(self, output_png: str) -> None:
        """
        For each assembly_type (FullQFT vs HalfQFT), plots:
            - size vs num_qubits
            - size vs depth
            - size vs width
            - size vs circuit_size
        on a 2x2 subplot grid, comparing FullQFT and HalfQFT on the same axes.
        Saves the figure to the specified PNG file.

        Args:
            output_png: Filename (including .png) where the plot will be saved.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() before plotting.")

        df = self.df.copy()
        # We expect columns: 'size', 'assembly_type', 'num_qubits', 'depth', 'width', 'circuit_size'
        metrics = ["num_qubits", "depth", "width", "circuit_size"]
        assembly_types = df["assembly_type"].unique()

        # Prepare subplots: 2 rows x 2 columns
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()  # flatten to iterate easily

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            ax.set_title(f"{metric.replace('_', ' ').title()} vs Size")
            ax.set_xlabel("Size")
            ax.set_ylabel(metric.replace("_", " ").title())

            # Group by size and assembly_type, then take mean of the metric
            grouped = df.groupby(["size", "assembly_type"])[metric].mean().unstack("assembly_type")

            # Plot each assembly_type on the same axes
            for asm in assembly_types:
                if asm in grouped.columns:
                    ax.plot(
                        grouped.index,
                        grouped[asm],
                        marker="o",
                        label=asm
                    )

            ax.legend()
            ax.grid(True)

        # Adjust layout and save to PNG
        plt.tight_layout()
        plt.savefig(output_png)
        plt.close(fig)
    
    def plot_gate_counts_histogram(self, output_png: str) -> None:
        """
        For each assembly_type (FullQFT vs HalfQFT), creates a bar chart where:
            - x-axis is 'size'
            - y-axis is gate count for each basis gate
          Each subplot corresponds to one assembly_type; bars for each size are
          grouped by basis gate, with a distinct color per gate.
        Saves the figure to the specified PNG file.

        Args:
            output_png: Filename (including .png) where the plot will be saved.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() before plotting.")

        df = self.df.copy()

        # Identify gate-count columns by excluding known metadata columns
        metadata_cols = {
            "size", "instance_id", "assembly_type",
            "num_qubits", "num_clbits", "depth", "width", "circuit_size"
        }
        gate_cols = [col for col in df.columns if col not in metadata_cols]

        assembly_types = df["assembly_type"].unique()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        for idx, asm in enumerate(assembly_types):
            ax = axes[idx]
            subset = df[df["assembly_type"] == asm]

            # Group by size and average each gate count across instances
            pivot = subset.groupby("size")[gate_cols].mean()

            # Create a grouped bar chart
            pivot.plot(
                kind="bar",
                ax=ax,
                colormap="tab20",
                width=0.8,
                legend=(idx == 1)  # show legend only on the right subplot
            )

            ax.set_title(f"{asm} ‐ Gate Counts by Size")
            ax.set_xlabel("Size")
            ax.set_ylabel("Average Gate Count")
            ax.grid(axis="y")

        # Adjust legend position to avoid overlap
        axes[1].legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            title="Basis Gates"
        )

        plt.tight_layout()
        plt.savefig(output_png)
        plt.close(fig)

    def plot_selected_gates_vs_size(self, output_png: str) -> None:
        """
        For the basis gates ['ecr', 'rz', 'sx', 'x'], generate a 2x2 subplot grid
        where each subplot corresponds to one of these gates. On each subplot,
        plot 'size' vs average gate count for both FullQFT and HalfQFT on the same axes.
        Saves the figure to the specified PNG file.

        Args:
            output_png: Filename (including .png) where the plot will be saved.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() before plotting.")

        df = self.df.copy()
        selected_gates = ["ecr", "rz", "sx", "x"]
        assembly_types = df["assembly_type"].unique()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, gate in enumerate(selected_gates):
            ax = axes[idx]
            ax.set_title(f"{gate.upper()} Count vs Size")
            ax.set_xlabel("Size")
            ax.set_ylabel("Average Count")

            # Ensure gate column exists
            if gate not in df.columns:
                raise KeyError(f"Gate '{gate}' not found in DataFrame columns.")

            # Group by size and assembly_type, then average the selected gate count
            grouped = df.groupby(["size", "assembly_type"])[gate].mean().unstack("assembly_type")

            for asm in assembly_types:
                if asm in grouped.columns:
                    ax.plot(
                        grouped.index,
                        grouped[asm],
                        marker="o",
                        label=asm
                    )

            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_png)
        plt.close(fig)

    def plot_transpiled_stats_vs_num_qubits(self, output_png: str) -> None:
        """
        For each assembly_type (FullQFT vs HalfQFT), plots num_qubits vs
        depth, width, circuit_size, size on a 2x2 grid. Saves to PNG.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() before plotting.")

        df = self.df.copy()
        # We want to plot num_qubits on x-axis and these metrics on y:
        metrics = ["depth", "width", "circuit_size", "size"]
        assembly_types = df["assembly_type"].unique()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            ax.set_title(f"{metric.replace('_', ' ').title()} vs Num Qubits")
            ax.set_xlabel("Num Qubits")
            ax.set_ylabel(metric.replace("_", " ").title())

            grouped = df.groupby(["num_qubits", "assembly_type"])[metric].mean().unstack("assembly_type")
            for asm in assembly_types:
                if asm in grouped.columns:
                    ax.plot(
                        grouped.index,
                        grouped[asm],
                        marker="o",
                        label=asm
                    )

            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_png)

    def plot_gate_counts_histogram_vs_num_qubits(self, output_png: str) -> None:
        """
        For each assembly_type, grouped bar chart of num_qubits vs gate counts for each basis gate.
        Saves to PNG.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() before plotting.")

        df = self.df.copy()
        metadata_cols = {
            "size", "instance_id", "assembly_type",
            "num_qubits", "num_clbits", "depth", "width", "circuit_size"
        }
        gate_cols = [col for col in df.columns if col not in metadata_cols]

        assembly_types = df["assembly_type"].unique()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        for idx, asm in enumerate(assembly_types):
            ax = axes[idx]
            subset = df[df["assembly_type"] == asm]
            pivot = subset.groupby("num_qubits")[gate_cols].mean()

            pivot.plot(
                kind="bar",
                ax=ax,
                colormap="tab20",
                width=0.8,
                legend=(idx == 1)
            )

            ax.set_title(f"{asm} ‐ Gate Counts by Num Qubits")
            ax.set_xlabel("Num Qubits")
            if idx == 0:
                ax.set_ylabel("Average Gate Count")
            ax.grid(axis="y")

        axes[1].legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            title="Basis Gates"
        )

        plt.tight_layout()
        plt.savefig(output_png)
        plt.close(fig)

    def plot_selected_gates_vs_num_qubits(self, output_png: str) -> None:
        """
        For gates ['ecr','rz','sx','x'], generate a 2x2 grid where each subplot
        corresponds to one gate. On each subplot, plot num_qubits vs average gate count
        for both FullQFT and HalfQFT. Saves to PNG.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() before plotting.")

        df = self.df.copy()
        selected_gates = ["ecr", "rz", "sx", "x"]
        assembly_types = df["assembly_type"].unique()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, gate in enumerate(selected_gates):
            ax = axes[idx]
            ax.set_title(f"{gate.upper()} Count vs Num Qubits")
            ax.set_xlabel("Num Qubits")
            ax.set_ylabel("Average Count")

            if gate not in df.columns:
                raise KeyError(f"Gate '{gate}' not found in DataFrame columns.")

            grouped = df.groupby(["num_qubits", "assembly_type"])[gate].mean().unstack("assembly_type")

            for asm in assembly_types:
                if asm in grouped.columns:
                    ax.plot(
                        grouped.index,
                        grouped[asm],
                        marker="o",
                        label=asm
                    )

            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_png)
        plt.close(fig)