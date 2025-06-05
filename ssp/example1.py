from solver import DGSSPSolver
from measurements import Results
import numpy as np

def main():
    A = [1, -3, 7]
    t = 5
    solver = DGSSPSolver(A, t, assembly_type="FullQFT")
    sum_neg = sum(a for a in A if a < 0)
    sum_pos = sum(a for a in A if a > 0)
    range_len = sum_pos - sum_neg + 1
    # n_sum must be an integer
    n_sum = int(np.ceil(np.log2(range_len)))
    n_ind = len(A)
    print(sum_neg,sum_pos,range_len)
    print(n_sum,n_ind)

    # Build the Grover circuit and print it
    qc = solver.solve(iterations=1)
    print(qc.draw())

    # Simulate:
    meas = Results(qc, shots=10**5)
    counts = meas.simulate()
    solutions = meas.instance_result(counts, A, t)

    #print("For instance:",{A},{t})
    print("Matching subsets and their probabilities:")
    for subset, prob in solutions:
        print(f"  subset={subset}, probability={prob:.4f}")

if __name__ == "__main__":
    main()