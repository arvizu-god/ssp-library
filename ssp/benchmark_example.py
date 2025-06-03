from solver import DGSSPSolver
from measurements import Results
from stats import Stats
import pandas as pd

#Example usage:
ds = {
    2: {
        1: {'A': [1, 2], 't': 1},
        2: {'A': [4, 5], 't': 4},
        3: {'A': [6, 7], 't': 13},
    },
    3: {
        1: {'A': [1, 2, 3], 't': 1},
        2: {'A': [5, 6, 9], 't': 11},
        3: {'A': [2, 7, 8], 't': 17},
    },
    4: {
        1: {'A': [1, 2, 3, 4], 't': 5},
        2: {'A': [3, 4, 5, 6], 't': 9},
        3: {'A': [2, 3, 10, 12], 't': 15},
    },
    5: {
        1: {'A': [1, 2, 3, 4, 5], 't': 7},
        2: {'A': [4, 5, 6, 7, 8], 't': 21},
        3: {'A': [1, 2, 3, 4, 5],  't': 7},
    },
    6: {
        1: {'A': [1, 2, 3, 4, 5, 6], 't': 1},
        2: {'A': [1, 2, 3, 4, 5, 6], 't': 2},
        3: {'A': [1, 2, 3, 4, 5, 6], 't': 10},
    },
    7: {
        1: {'A': [1, 2, 3, 4, 5, 6, 7], 't': 1},
        2: {'A': [1, 2, 3, 4, 5, 6, 7], 't': 2},
        3: {'A': [1, 2, 3, 4, 5, 6, 7], 't': 14},
    },
    8: {
        1: {'A': [1, 2, 3, 4, 5, 6, 7, 8], 't': 1},
        2: {'A': [1, 2, 3, 4, 5, 6, 7, 8], 't': 2},
        3: {'A': [1, 2, 3, 4, 5, 6, 7, 8], 't': 18},
    },
    9: {
        1: {'A': list(range(1, 10)), 't': 1},
        2: {'A': list(range(1, 10)), 't': 2},
        3: {'A': list(range(1, 10)), 't': 22},
    },
    10: {
        1: {'A': list(range(1, 11)), 't': 1},
        2: {'A': list(range(1, 11)), 't': 2},
        3: {'A': list(range(1, 11)), 't': 27},
    },
    11: {
        1: {'A': list(range(1, 12)), 't': 1},
        2: {'A': list(range(1, 12)), 't': 2},
        3: {'A': list(range(1, 12)), 't': 33},
    },
    12: {
        1: {'A': list(range(1, 13)), 't': 1},
        2: {'A': list(range(1, 13)), 't': 2},
        3: {'A': list(range(1, 13)), 't': 39},
    },
}
#Then in your top‐level script you could do:
stats_runner = Stats(ds, shots=10**5)
df = stats_runner.run()
df_display = df.reset_index()
print(df_display)
stats_runner.save_to_csv(df, "ssp_benchmark_results.csv")
t_df=stats_runner.run_transpiled()
tdf_display=t_df.reset_index()
print(tdf_display)
stats_runner.save_transpiled_to_csv(t_df,"ssp_transpiled_benchmark_results.csv")