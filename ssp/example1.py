from solver import DGSSPSolver
from measurements import Results

def main():
    A = [1, -3, 7]
    t = 5
    solver = DGSSPSolver(A, t, assembly_type="FullQFT")

    # Build the Grover circuit and print it
    qc = solver.solve(iterations=1)
    print(qc.draw('mpl'))

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