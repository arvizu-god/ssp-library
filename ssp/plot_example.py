from visualization import Plots

# 1. Instantiate with path to CSV
plots = Plots("ssp_transpiled_benchmark_results.csv")

# 2. Load the data
df = plots.load_data()


# Original style: size on x-axis
plots.plot_transpiled_stats("stats_vs_size.png")
plots.plot_gate_counts_histogram("histogram_vs_size.png")
plots.plot_selected_gates_vs_size("selected_gates_vs_size.png")

# New style: num_qubits on x-axis
plots.plot_transpiled_stats_vs_num_qubits("stats_vs_num_qubits.png")
plots.plot_gate_counts_histogram_vs_num_qubits("histogram_vs_num_qubits.png")
plots.plot_selected_gates_vs_num_qubits("selected_gates_vs_num_qubits.png")