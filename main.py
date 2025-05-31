#!/usr/bin/env python
import random
from typing import Callable, Generator
import qiskit
from qiskit.circuit import Parameter, QuantumCircuit

K = 20
M = 20
N = 100
L_MAX = 5
QUBITS = 4


def gate_combinations(qubits: int):
    if qubits == 0:
        yield []
    else:
        # combinations for the first N-1 qubits
        sub_combinations = gate_combinations(qubits - 1)
        for combination in sub_combinations:
            yield combination + [0]
            yield combination + [1]
            yield combination + [2]

            for offset in range(1, len(combination) + 1):
                if combination[-offset] == 0 and all(
                    combination[-offset + o] != o + 2 for o in range(1, offset + 1)
                ):
                    yield combination + [offset + 2]


def create_qiskit_QC(
    instructions: list[list[int]],
) -> Callable[[list[int], list[int]], QuantumCircuit]:
    qubits = len(instructions[0])

    xparams = [Parameter(f"x-{i}") for i in range(qubits)]

    qc = qiskit.QuantumCircuit(qiskit.QuantumRegister(qubits))
    qc.h([i for i in range(qubits)])
    for qbit in range(qubits):
        qc.rx(xparams[qbit], qbit)

    iparams = []

    for layer in instructions:
        for qubit, operation in enumerate(layer):
            if operation == 0:
                pass
            if operation == 1:
                qc.h(qubit)
            if operation == 2:
                iparams.append(Parameter(f"instr-{len(iparams)}"))
                qc.rz(iparams[-1], qubit)
            if operation == 3:
                qc.cx(qubit, qubit - 1)
            if operation == 4:
                qc.cx(qubit, qubit - 2)
            if operation == 5:
                qc.cx(qubit, qubit - 3)

    def filled_qc(values: list[int], parameters: list[int]):
        return qc.assign_parameters(
            {f"x-{i}": val for i, val in enumerate(values)}
        ).assign_parameters({f"instr-{i}": val for i, val in enumerate(parameters)})

    return filled_qc


# the circuit from fig 1 from the paper
initial_circuit = []


optimal_quantum_circuits = [([], 1000000000) for _ in range(K)]

for i in range(L_MAX):
    new_circs = []
    for circ, _bic_score in optimal_quantum_circuits:
        for combination in gate_combinations(QUBITS):
            new_circ = circ + [combination]
            # TODO: compute quantum kernel

            # TODO: train an SVM model

            # TODO: convert outputs to probabilistic predictions

            # TODO: calculate BIC with validation set
            bic_score = 0

            new_circs.append((new_circ, bic_score))

    # TODO: use the comment instead of shuffle once bic_score is fixed
    random.shuffle(new_circs)
    # new_circs.sort(key=lambda entry: entry[1])
    optimal_quantum_circuits = new_circs[:K]

    # print currently optimal circuits
    for circ in optimal_quantum_circuits:
        print(circ)
        print(create_qiskit_QC(circ[0])([], []))
