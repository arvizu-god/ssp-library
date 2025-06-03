from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister,transpile
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_aer import AerSimulator, QasmSimulator
from typing import List, Dict, Tuple

from solver import DGSSPSolver

class Results:
    def __init__(self, qc: QuantumCircuit, shots: int):
        self.qc = qc
        self.shots = shots

    def simulate(self) -> Dict[str, int]:
        """
        Run the circuit (self.qc) on AerSimulator and return the counts dictionary.
        """
        simulator = AerSimulator(shots=self.shots)
        transpiled = transpile(self.qc, simulator)
        result = simulator.run(transpiled).result()
        counts = result.get_counts()
        return counts 

    def instance_result(
        self,
        counts: Dict[str, int],
        A: List[int],
        t: int
    ) -> List[Tuple[List[int], float]]:
        """
        Given a Qiskit counts dict, reconstruct those subsets of A whose
        bit‐pattern measurement sums to t. Returns a list of (subset, probability).
        """
        total_shots = sum(counts.values())

        # Sort (bitstring → raw count) pairs by descending count
        sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

        answer_subsets = []
        for state, raw_count in sorted_items:
            # 'state' is e.g. '010' if n_ind=3. We want LSB-first,
            # so reverse the entire string.
            subset_qubits = list(reversed(state))  

            # Build the subset of A where the measured bit == '1'
            subset = [A[i] for i, bit in enumerate(subset_qubits) if bit == '1']

            if sum(subset) == t:
                probability = raw_count / total_shots
                answer_subsets.append((subset, probability))
            # Don’t break—lower‐count bitstrings might also sum to t
            # else:
            #     continue

        return answer_subsets