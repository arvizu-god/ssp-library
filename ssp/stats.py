import time
import pandas as pd
from math import log, sqrt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister,transpile
from qiskit_aer import AerSimulator, QasmSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from typing import List, Dict, Tuple


from solver import DGSSPSolver
from measurements import Results

class Stats:
    """
    Benchmarking class for running SSP instances with all available assembly types,
    simulating the resulting circuits, and collecting performance metrics, both
    for ideal and for transpiled circuits on a fixed set of basis gates.
    """

    def __init__(self, ds: Dict[int, Dict[int, Dict[str, List[int]]]], shots: int = 10**5):
        """
        Args:
            ds: A nested dictionary of the form
                {
                    size: {
                        inst_id: {'A': [...], 't': ...},
                        ...
                    },
                    ...
                }
            shots: Number of QASM shots to run when simulating each circuit.
        """
        self.ds = ds
        self.shots = shots
        self.assembly_types = ["FullQFT", "HalfQFT"]
        # Fixed basis gates for transpilation
        self.basis_gates = [
            "ecr", "id", "delay", "measure", "reset",
            "rz", "sx", "x", "if_else", "for_loop", "switch_case"
        ]

    def run(self) -> pd.DataFrame:
        """
        Executes the benchmark loop over every instance in `ds` and over all assembly types.
        Returns a pandas DataFrame indexed by (size, instance_id, assembly_type),
        containing circuit metrics, simulation metrics, and gate‐counts.
        """
        sim = AerSimulator(shots=self.shots)
        records = []

        for size, insts in self.ds.items():
            for inst_id, params in insts.items():
                A, t = params["A"], params["t"]
                n = len(A)

                for assembly in self.assembly_types:
                    # 1) Build & solve circuit for this assembly type
                    solver = DGSSPSolver(A, t, assembly_type=assembly)
                    qc = solver.solve(iterations=1)

                    # 2) Basic circuit statistics
                    num_qubits = qc.num_qubits
                    num_clbits = qc.num_clbits
                    depth = qc.depth()
                    width = qc.width()
                    circ_size = qc.size()

                    # 3) Simulate with AerSimulator (QASM)
                    qc_meas = qc.copy()
                    transpiled = transpile(qc_meas, sim)

                    start_time = time.time()
                    job = sim.run(transpiled)
                    result = job.result()
                    exec_time = time.time() - start_time

                    counts: Dict[str, int] = result.get_counts()

                    # 4) Compute success probability using Results.instance_result
                    results_helper = Results(qc, shots=self.shots)
                    solutions_with_prob = results_helper.instance_result(counts, A, t)
                    success_prob = sum(prob for (_, prob) in solutions_with_prob)
                    solutions = [subset for (subset, _) in solutions_with_prob]

                    # 5) Decomposed gate counts
                    gate_counts = qc.decompose().count_ops()

                    # 6) Assemble a single record
                    record = {
                        "size": size,
                        "instance_id": inst_id,
                        "assembly_type": assembly,
                        "solutions": solutions,
                        "num_qubits": num_qubits,
                        "num_clbits": num_clbits,
                        "depth": depth,
                        "width": width,
                        "circuit_size": circ_size,
                        "execution_time_sec": exec_time,
                        "success_probability": success_prob,
                        **{str(g): int(c) for g, c in gate_counts.items()},
                    }
                    records.append(record)

        df = pd.DataFrame(records).set_index(["size", "instance_id", "assembly_type"])
        return df

    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save the benchmark DataFrame to a CSV file.

        Args:
            df: The pandas DataFrame returned by run().
            filepath: Path (including filename) where the CSV should be written.
        """
        df.to_csv(filepath)

    def run_transpiled(self) -> pd.DataFrame:
        """
        For each instance and each assembly type, transpile the DGSSPSolver circuit
        into the fixed basis gates and collect metrics on the transpiled circuits.
        Returns a DataFrame indexed by (size, instance_id, assembly_type), containing:
            - num_qubits, num_clbits, depth, width, circuit_size
            - counts of each gate in basis_gates (zero if absent)
        """
        records = []

        for size, insts in self.ds.items():
            for inst_id, params in insts.items():
                A, t = params["A"], params["t"]

                for assembly in self.assembly_types:
                    # Build the base circuit
                    solver = DGSSPSolver(A, t, assembly_type=assembly)
                    qc = solver.solve(iterations=1)

                    # Transpile into fixed basis gates
                    qc_tp = transpile(qc, basis_gates=self.basis_gates)

                    # Metrics on the transpiled circuit
                    num_qubits = qc_tp.num_qubits
                    num_clbits = qc_tp.num_clbits
                    depth = qc_tp.depth()
                    width = qc_tp.width()
                    circ_size = qc_tp.size()

                    # Count ops (only basis gates will appear after transpilation)
                    gate_counts = qc_tp.count_ops()

                    # Assemble record
                    record = {
                        "size": size,
                        "instance_id": inst_id,
                        "assembly_type": assembly,
                        "num_qubits": num_qubits,
                        "num_clbits": num_clbits,
                        "depth": depth,
                        "width": width,
                        "circuit_size": circ_size,
                        **{str(g): int(gate_counts.get(g, 0)) for g in self.basis_gates},
                    }
                    records.append(record)

        df_tp = pd.DataFrame(records).set_index(["size", "instance_id", "assembly_type"])
        return df_tp

    def save_transpiled_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save the transpiled‐circuit metrics DataFrame to a CSV file.

        Args:
            df: The pandas DataFrame returned by run_transpiled().
            filepath: Path (including filename) where the CSV should be written.
        """
        df.to_csv(filepath)