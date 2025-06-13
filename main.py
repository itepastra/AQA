#!/usr/bin/env python
import itertools
import json
import numpy as np
from typing import List
import pennylane as qml
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
from qiskit_machine_learning.datasets import ad_hoc_data
import matplotlib.pyplot as plt
import matplotlib
import random
from tqdm.notebook import tqdm
from skopt import gp_minimize
from skopt.space import Real
from sklearn.metrics import accuracy_score, f1_score, classification_report
import seaborn as sns
import pandas as pd
from collections import Counter
from pqdm.processes import pqdm

K = 5  # Top circuits to keep per iteration
N = 100  # Data size
L_MAX = 4  # Max circuit depth
QUBITS = 3  # Number of qubits
JOBS = 20
seed = 42

projector = np.zeros((2**QUBITS, 2**QUBITS))
projector[0, 0] = 1


def gate_combinations_sub(qubits: int, previous_layer: tuple[int]):
    if qubits == 0:
        yield ()
    else:
        for combination in gate_combinations_sub(qubits - 1, previous_layer):
            yield combination + (0,)
            if previous_layer[qubits - 1] != 1:
                yield combination + (1,)
            if previous_layer[qubits - 1] != 2:
                yield combination + (2,)
            for offset in range(1, len(combination) + 1):
                if (
                    combination[-offset] == 0
                    and all(
                        combination[-offset + o] != o + 2 for o in range(1, offset + 1)
                    )
                    and previous_layer[qubits - 1] != offset + 2
                ):
                    yield combination + (offset + 2,)


def gate_combinations(qubits, previous_layer):
    # skip the 1st output (0 everywhere) so we actually do
    return list(gate_combinations_sub(qubits, previous_layer))[3:]


def create_pennylane_circuit(instructions: List[List[int]]):
    dev = qml.device("default.qubit", wires=QUBITS)

    @qml.qnode(dev)
    def circuit(xparams=[], yparams=[]):
        for q in range(QUBITS):
            qml.Hadamard(wires=q)
            qml.RZ(xparams[q], wires=q)

        idx = 0
        for layer in instructions:
            for qbit, op in enumerate(layer):
                if op == 0:
                    continue
                elif op == 1:
                    qml.Hadamard(wires=qbit)
                elif op == 2:
                    qml.RZ(yparams[idx], wires=qbit)
                    idx += 1
                elif op >= 3:
                    qml.CNOT(wires=[qbit, qbit - op + 2])
        return qml.state()

    return circuit


def build_kernel_fn(gate_layers, rz_params):
    dev = qml.device("default.qubit", wires=QUBITS)
    dev2 = qml.device("default.qubit", wires=QUBITS)

    def apply_circuit(x):
        idx = 0
        for q in range(QUBITS):
            qml.Hadamard(wires=q)
            qml.RZ(x[q], wires=q)
        for layer in gate_layers:
            for qbit, op in enumerate(layer):
                if op == 0:
                    continue
                elif op == 1:
                    qml.Hadamard(wires=qbit)
                elif op == 2:
                    qml.RZ(rz_params[idx], wires=qbit)
                    idx += 1
                elif op >= 3:
                    qml.CNOT(wires=[qbit, qbit - op + 2])

    @qml.qnode(dev)
    def kernel_qnode(x1, x2):
        apply_circuit(x1)
        qml.adjoint(apply_circuit)(x2)
        # return qml.probs(wires=0)
        # return qml.expval(qml.PauliZ(wires=range(QUBITS)))
        return qml.expval(qml.Hermitian(projector, wires=range(QUBITS)))

    def kernel_fn(X1, X2):
        K = np.zeros((len(X1), len(X2)))
        for i in range(len(X1)):
            for j in range(len(X2)):
                K[i, j] = kernel_qnode(X1[i], X2[j])
        return K

    return kernel_fn


def compute_information_criteria(y_true, y_prob, num_params):
    y_pred = (y_prob >= 0.5).astype(int)
    n = len(y_true)
    loglik = -log_loss(y_true, y_prob, normalize=False)
    aic = 2 * num_params - 2 * loglik
    bic = num_params * np.log(n) - 2 * loglik

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    class_accuracies = {
        f"class_{cls}": report[cls]["recall"] for cls in report if cls in ("0", "1")
    }

    return aic, bic, acc, class_accuracies, f1


gap = 0.3

# Load data
x_train_raw, y_train_raw, x_test_raw, y_test_raw = ad_hoc_data(
    training_size=N // 2, test_size=N // 2, n=QUBITS, gap=gap, one_hot=False
)

X_test, y_test, tempx, tempy = ad_hoc_data(
    training_size=10 // 2, test_size=0, n=QUBITS, gap=gap, one_hot=False
)

x = np.vstack([x_train_raw, x_test_raw])
y = np.hstack([y_train_raw, y_test_raw])
x, y = shuffle(x, y, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.3, random_state=seed
)


def compute_test_values(circ, rz, model):
    num_rz = len(rz)
    try:
        kernel_fn = build_kernel_fn(circ, rz)
        K_test = kernel_fn(X_test, x_train)
        y_prob = model.predict_proba(K_test)[:, 1]
        aic, bic, acc, class_accs, f1 = compute_information_criteria(
            y_test, y_prob, num_rz
        )
        return acc, bic, acc, class_accs, f1
    except Exception as e:
        print(f"Test evaluation error: {e}")
        return None, None, None, None, None


best_circuit_arr = []

for m in [1, 5, 10, 15, 20]:
    optimal_circuits = []
    for depth in tqdm(range(1, L_MAX + 1), desc="Depth", position=3, leave=False):
        base_circuits = (
            [(c[0], c[1]) for c in optimal_circuits]
            if optimal_circuits
            else [([tuple(0 for _ in range(QUBITS))], [])]
        )

        # Stage 1: Structure search (fixed/random RZ)
        def calculate_combo(inp):
            base, base_rz, combo = inp
            new_circ = base + [combo]
            num_rz = sum(layer.count(2) for layer in new_circ)
            # print(f"Trying circuit {new_circ} with {num_rz} RZs")
            new_rz_count = combo.count(2)
            new_rz = np.random.uniform(-np.pi, np.pi, size=new_rz_count).tolist()
            dummy_rz = base_rz + new_rz

            try:
                kernel_fn = build_kernel_fn(new_circ, dummy_rz)
                K_train = kernel_fn(x_train, x_train)
                model = SVC(kernel="precomputed", probability=True)
                model.fit(K_train, y_train)

                K_val = kernel_fn(x_val, x_train)
                y_prob = model.predict_proba(K_val)[:, 1]

                aic, bic, acc, class_accs, f1 = compute_information_criteria(
                    y_val, y_prob, num_rz
                )
                print(f"combo {combo} has BIC {bic}")
                return (new_circ, dummy_rz, aic, bic, acc, class_accs, f1, model)
            except Exception as e:
                print(f"Structure error: {e}")
                return None

        result = pqdm(
            [
                (base, rz, combo)
                for base, rz in base_circuits
                for combo in gate_combinations(QUBITS, base[-1])
            ],
            calculate_combo,
            n_jobs=JOBS,
            desc="Gate Combinations",
            position=1,
            leave=False,
        )
        stage1_candidates = []
        for thing in result:
            stage1_candidates.append(thing)

        # Pick top K by BIC
        stage1_candidates.sort(key=lambda x: x[3])
        # top_k = stage1_candidates[:K]
        param_circuits = [item for item in stage1_candidates]
        # top_m = param_circuits[:M]
        # Stage 2: Parameter optimization on top M circuits
        stage2_optimized = []

        def parameter_optimization(values):
            circ, init_rz, aic, bic, acc, class_accs, f1, model1 = values
            num_rz = len(init_rz)
            if num_rz == 0:
                return (circ, [], aic, bic, acc, class_accs, f1, model1)

            def objective(params):
                try:
                    kernel_fn = build_kernel_fn(circ, params)
                    K_train = kernel_fn(x_train, x_train)
                    model = SVC(kernel="precomputed", probability=True)
                    model.fit(K_train, y_train)
                    K_val = kernel_fn(x_val, x_train)
                    y_prob = model.predict_proba(K_val)[:, 1]
                    aic, bic, acc, class_accs, f1 = compute_information_criteria(
                        y_val, y_prob, num_rz
                    )
                    return bic
                except Exception:
                    return 1e6

            space = [Real(-np.pi, np.pi) for _ in range(num_rz)]
            result = gp_minimize(objective, space, n_calls=50, random_state=seed)
            best_params = result.x

            try:
                kernel_fn = build_kernel_fn(circ, best_params)
                K_train = kernel_fn(x_train, x_train)
                model = SVC(kernel="precomputed", probability=True)
                model.fit(K_train, y_train)
                K_val = kernel_fn(x_val, x_train)
                y_prob = model.predict_proba(K_val)[:, 1]
                aic, bic, acc, class_accs, f1 = compute_information_criteria(
                    y_val, y_prob, num_rz
                )
                return (circ, best_params, aic, bic, acc, class_accs, f1, model)

            except Exception as e:
                print(f"Final model error: {e}")
                return None

        stage2_optimized = [
            x
            for x in pqdm(
                param_circuits[:m],
                parameter_optimization,
                n_jobs=JOBS,
                position=2,
                leave=False,
                desc="Optimizing parameters",
            )
            if x is not None
        ]

        # Add remaining K-M circuits (unoptimized) + optimized ones
        optimal_circuits = stage2_optimized
        optimal_circuits.sort(key=lambda x: x[3])  # sort by BIC
        optimal_circuits = optimal_circuits[:K]  # keep top K

        aic, bic, acc, class_accs, f1 = compute_test_values(
            optimal_circuits[0][0], optimal_circuits[0][1], optimal_circuits[0][7]
        )

        best_circuit_arr.append(
            (optimal_circuits[0] + (m, depth) + (aic, bic, acc, class_accs, f1))
        )

        with open("data.json", "w+") as f:
            json.dump(
                [
                    (
                        m,
                        depth,
                        circ,
                        best_params,
                        aic,
                        bic,
                        acc,
                        class_accs,
                        f1,
                        aic_test,
                        bic_test,
                        acc_test,
                        class_accs_test,
                        f1_test,
                    )
                    for (
                        circ,
                        best_params,
                        aic,
                        bic,
                        acc,
                        class_accs,
                        f1,
                        model,
                        m,
                        depth,
                        aic_test,
                        bic_test,
                        acc_test,
                        class_accs_test,
                        f1_test,
                    ) in best_circuit_arr
                ],
                f,
            )
