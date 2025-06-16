#!/usr/bin/env python
import pennylane as qml
import matplotlib.pyplot as plt
import matplotlib

QUBITS = 3


def create_pennylane_circuit(instructions: list[list[int]]):
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
                    qml.RZ(yparams[idx] * xparams[qbit], wires=qbit)
                    idx += 1
                elif op == 3:
                    qml.RX(yparams[idx] * xparams[qbit], wires=qbit)
                    idx += 1
                elif op == 4:
                    qml.RY(yparams[idx] * xparams[qbit], wires=qbit)
                    idx += 1
                elif op == 5:
                    qml.CNOT(wires=[qbit, qbit - 1])
        return qml.state()

    return circuit


matplotlib.use("tkagg")

circ = create_pennylane_circuit([[0, 0, 0]])
print(qml.draw(circ([1, 2, 3])))
fig, ax = qml.draw_mpl(circ)([1, 2, 3])
plt.show()
