from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister,transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram, plot_distribution
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_aer import AerSimulator, QasmSimulator

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class DGSSPSolver:
    """
    Draper‐Grover Subset‐Sum solver.
    Encapsulates creation of sum/sub gates, oracle, diffuser,
    assembly of one SSP step, and full Grover loop + measurement.
    """
    def __init__(self, A: List[int], t: int, assembly_type:str):
        self.A = A
        self.t = t
        self.assembly_type=assembly_type

        # compute negative/positive sums correctly
        self.sum_neg = sum(a for a in A if a < 0)
        self.sum_pos = sum(a for a in A if a > 0)
        self.range_len = self.sum_pos - self.sum_neg + 1

        # n_sum must be an integer
        self.n_sum = int(np.ceil(np.log2(self.range_len)))
        self.n_ind = len(self.A)

        # prebuild QFT and inverse-QFT gates on n_sum qubits
        self.qft = QFT(self.n_sum, do_swaps=False).to_gate(label="QFT")
        self.iqft = QFT(self.n_sum, do_swaps=False).inverse().to_gate(label="iQFT")

    def instance(self):
        return (self.A,self.t)

    def sum_gate(self) -> Gate:
        """Controlled‐phase adder for A into the sum register."""
        ind = QuantumRegister(self.n_ind, name="i")
        summ = QuantumRegister(self.n_sum, name="s")
        qc = QuantumCircuit(ind, summ, name="SumGate")
        modulus = 2 ** self.n_sum

        for k, a in enumerate(self.A):
            a_mod = a % modulus
            for j in range(self.n_sum):
                phi = (2 * np.pi * a_mod) / (2 ** (j + 1))
                qc.cp(phi, ind[k], summ[j])

        return qc.to_gate(label="SumGate")

    def sub_gate(self) -> Gate:
        """Controlled‐phase subtractor for A from the sum register."""
        ind = QuantumRegister(self.n_ind, name="i")
        summ = QuantumRegister(self.n_sum, name="s")
        qc = QuantumCircuit(ind, summ, name="SubGate")
        modulus = 2 ** self.n_sum

        for k, a in enumerate(self.A):
            neg = (-a) % modulus
            for j in range(self.n_sum):
                phi = (2 * np.pi * neg) / (2 ** (j + 1))
                qc.cp(phi, ind[k], summ[j])

        return qc.to_gate(label="SubGate")

    def oracle_gate(self) -> Gate:
        """Phase oracle that flips phase of |s⟩ = |t⟩."""
        summ = QuantumRegister(self.n_sum, name="s")
        qc = QuantumCircuit(summ, name="OracleGate")

        # flip the bits where t has a 0
        for j in range(self.n_sum):
            if ((self.t >> j) & 1) == 0:
                qc.x(summ[j])

        # multi-controlled Z via H–MCX–H
        qc.h(summ[-1])
        qc.mcx(summ[:-1], summ[-1], mode="noancilla")
        qc.h(summ[-1])

        # unflip
        for j in range(self.n_sum):
            if ((self.t >> j) & 1) == 0:
                qc.x(summ[j])

        return qc.to_gate(label="OracleGate")

    def grover_diffuser(self) -> Gate:
        """Standard n_ind-qubit Grover diffuser on the index register."""
        ind = QuantumRegister(self.n_ind, name="i")
        qc = QuantumCircuit(ind, name="GroverDiffuser")

        # H–X on all
        for qb in ind:
            qc.h(qb)
            qc.x(qb)

        # multi-controlled Z
        qc.h(ind[-1])
        qc.mcx(ind[:-1], ind[-1], mode="noancilla")
        qc.h(ind[-1])

        # X–H on all
        for qb in ind:
            qc.x(qb)
            qc.h(qb)

        return qc.to_gate(label="GroverDiffuser")

    def assembly(self) -> QuantumCircuit:
        """
        There are two type of assembly options.
        FullQFT means that both QFT and IQFT will be applied when generating the search space and when un-entangling the sum register.
        HalfQFT means that a Hadamard operation will be applied instead of the initial QFT.
        """
        ind = QuantumRegister(self.n_ind, name="i")
        summ = QuantumRegister(self.n_sum, name="s")
        qc = QuantumCircuit(ind, summ, name="DGSSP_Step")

        if self.assembly_type=='FullQFT':
            # add
            qc.append(self.qft, summ[:])
            qc.append(self.sum_gate(), ind[:] + summ[:])
            qc.append(self.iqft, summ[:])

            # oracle
            qc.append(self.oracle_gate(), summ[:])

            # subtract
            qc.append(self.qft, summ[:])
            qc.append(self.sub_gate(), ind[:] + summ[:])
            qc.append(self.iqft, summ[:])

            return qc

        elif self.assembly_type=='HalfQFT':
            # add
            qc.h(summ[:])
            qc.append(self.sum_gate(), ind[:] + summ[:])
            qc.append(self.iqft, summ[:])

            # oracle
            qc.append(self.oracle_gate(), summ[:])

            # subtract
            qc.append(self.iqft, summ[:]).inverse()
            qc.append(self.sub_gate(), ind[:] + summ[:])
            qc.h(summ[:])

            return qc
        else:
            return qc

    def solve(self, iterations: int) -> QuantumCircuit:
        """
        Build the full Grover circuit and measure indices.
        Returns a QuantumCircuit ready to run (with measurements).
        """
        ind = QuantumRegister(self.n_ind, name="i")
        summ = QuantumRegister(self.n_sum, name="s")
        creg = ClassicalRegister(self.n_ind, name="c")
        qc = QuantumCircuit(ind, summ, creg, name="DGSSPSolve")

        # initialize index register
        qc.h(ind)
        qc.barrier()

        # Grover iterations
        step_gate = self.assembly().to_gate(label="DGSSP_Step")
        diffuser = self.grover_diffuser()
        for _ in range(iterations):
            qc.append(step_gate, ind[:] + summ[:])
            qc.barrier()
            qc.append(diffuser, ind[:])
            qc.barrier()

        # measurement
        qc.measure(ind, creg)
        return qc
