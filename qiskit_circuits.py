"""Qiskit circuits for CHSH and EWL, cross-checked against the numpy model in qgames.py.
Tested on qiskit 2.6.1 / qiskit-aer 0.18.2.

Traps (all fail SILENTLY):
  1. Bitstrings are little-endian: key = 'Bob Alice'. CD = '10', DC = '01'.
  2. Never transpile noisy EWL circuits at optimization_level >= 1: J.U.J^dag gets merged and the noise vanishes.
  3. Measuring along angle a needs ry(-a), not ry(+a).
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYYGate, UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from qgames import U, C, D, Q, PA, PB, ewl as ewl_numpy

# ---------- EWL ----------
def ewl_circuit(UA, UB, g, measure=False):
    qc = QuantumCircuit(2)
    qc.append(RYYGate(g), [0, 1])             # J = exp(i g DxD/2) == RYY(+g), since DxD = -YxY
    qc.append(UnitaryGate(UA, label="A"), [0])  # Alice = qubit 0
    qc.append(UnitaryGate(UB, label="B"), [1])  # Bob   = qubit 1
    qc.append(RYYGate(g).inverse(), [0, 1])   # J^dag
    if measure:
        qc.measure_all()
    return qc

def ewl_payoffs(UA, UB, g):
    p = Statevector(ewl_circuit(UA, UB, g)).probabilities_dict()
    pr = np.array([p.get(k, 0) for k in ("00", "10", "01", "11")])  # CC, CD, DC, DD
    return pr @ PA, pr @ PB

# ---------- CHSH ----------
def chsh_circuit(a, b):
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    qc.ry(-a, 0); qc.ry(-b, 1)
    qc.measure_all()
    return qc

def chsh_sampled(shots=100_000, seed=1):
    pairs = [(0, np.pi / 4), (0, -np.pi / 4), (np.pi / 2, np.pi / 4), (np.pi / 2, -np.pi / 4)]
    res = StatevectorSampler(seed=seed).run([chsh_circuit(a, b) for a, b in pairs], shots=shots).result()
    E = []
    for r in res:
        c = r.data.meas.get_counts()
        E.append(sum((1 if k.count("1") % 2 == 0 else -1) * n for k, n in c.items()) / shots)
    S = E[0] + E[1] + E[2] - E[3]
    sigma = np.sqrt(sum(1 - e ** 2 for e in E) / shots)
    return S, sigma

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for _ in range(20):  # random strategies + gamma: Qiskit must equal numpy
        UA, UB = U(*rng.uniform(0, np.pi, 2)), U(*rng.uniform(0, np.pi, 2))
        g = rng.uniform(0, np.pi / 2)
        assert np.allclose(ewl_payoffs(UA, UB, g), ewl_numpy(UA, UB, g))
    assert np.allclose(ewl_payoffs(Q, Q, np.pi / 2), (3, 3))
    assert np.allclose(ewl_payoffs(C, D, 0), (0, 5))
    S, s = chsh_sampled()
    assert abs(S - 2 * np.sqrt(2)) < 5 * s, (S, s)
    print(f"qiskit == numpy on 20 random EWL cases; CHSH S = {S:.4f} +- {s:.4f}")
