#!/usr/bin/env python
from typing import Callable
import numpy as np
import pennylane as qml
from pennylane import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import log_loss
from itertools import product, combinations
import copy
from skopt import gp_minimize
from skopt.space import Real
from tqdm.notebook import tqdm
import time
import matplotlib.pyplot as plt
from termcolor import colored
from qiskit_machine_learning.datasets import ad_hoc_data
from sklearn.utils import shuffle
from scipy.special import expit

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


def create_pennylane_circuit(
    instructions: list[list[int]],
):
    qubits = len(instructions[0])
    dev = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev)
    def circuit(xparams=[], yparams=[]):
        for qbit in range(qubits):
            qml.H(wires=qbit)
            qml.RX(xparams[qbit], wires=qbit)

        idx = 0
        for layer in instructions:
            for qbit, operation in enumerate(layer):
                if operation == 0:
                    pass
                elif operation == 1:
                    qml.H(wires=qbit)
                elif operation == 2:
                    qml.RZ(yparams[idx], wires=qbit)
                    idx += 1
                elif operation >= 3:
                    qml.CNOT(wires=[qbit, qbit - operation + 2])

    return circuit


# the circuit from fig 1 from the paper
initial_circuit = []


optimal_quantum_circuits = [([], 0)]

for circ in gate_combinations(QUBITS):
    print(circ)
    xparams = [0 for _ in range(1000)]
    yparams = [0 for _ in range(1000)]
    drawer = qml.draw(create_pennylane_circuit([circ]))
    print(drawer(xparams, yparams))

exit()

for i in range(L_MAX):
    new_circs = []
    for circ, _bic_score in optimal_quantum_circuits:
        for combination in gate_combinations(QUBITS):
            new_circ = circ + [combination]

            # Compute BIC
            bic_score = compute_bic(probs, y_val, num_params)

            new_circs.append((new_circ, bic_score))

    new_circs.sort(key=lambda entry: entry[1])
    optimal_quantum_circuits = new_circs[:K]

    # print currently optimal circuits
    for circ in optimal_quantum_circuits:
        print(circ)
        print(create_pennylane_circuit(circ[0])([], []))
