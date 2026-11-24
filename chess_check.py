"""Quantum chess split/merge (Cantwell, arXiv:1906.05836) on three squares s, a, b under dephasing.
Qubit = occupancy of a square. W = iSWAP, V = sqrt(iSWAP). Split s -> (a, b): W_sb V_sa. Merge: V_sa W_sb.
Checks: the split gives i(|001>+|010>)/sqrt2; the merge recovers |100> with probability (1+k)/2 for the
coherent split and 1/2 for the 50/50 incoherent mixture, k=(1-p)^2; target negativity k/2."""
import numpy as np
from functools import reduce
from qgames import kraus

W = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
h = 1 / np.sqrt(2)
V = np.array([[1, 0, 0, 0], [0, h, 1j * h, 0], [0, 1j * h, h, 0], [0, 0, 0, 1]])
assert np.allclose(V @ V, W) and np.allclose(V.conj().T @ V, np.eye(4))

def on(G, i, j):   # two-qubit gate G on qubits (i, j) of (s, a, b), qubit 0 most significant
    perm = [i, j] + [q for q in range(3) if q not in (i, j)]
    full = np.kron(G, np.eye(2)).reshape([2] * 6)
    inv = np.argsort(perm)
    return full.transpose(list(inv) + [3 + x for x in inv]).reshape(8, 8)

ket = lambda bits: np.eye(8)[int(bits, 2)]
SPLIT, MERGE = on(W, 0, 2) @ on(V, 0, 1), on(V, 0, 1) @ on(W, 0, 2)
psi = SPLIT @ ket("100")
assert np.allclose(psi, 1j * (ket("001") + ket("010")) / np.sqrt(2))
assert np.allclose(MERGE @ psi, -ket("100"))                       # merge undoes the split, phase -1
psi_perp = 1j * (ket("001") - ket("010")) / np.sqrt(2)
assert np.allclose(MERGE @ psi_perp, -1j * ket("010"))             # orthogonal branch state never returns
inc0 = (np.outer(ket("001"), ket("001")) + np.outer(ket("010"), ket("010"))) / 2
assert np.allclose(inc0, (np.outer(psi, psi.conj()) + np.outer(psi_perp, psi_perp.conj())) / 2)

def noisy(rho, p):   # same dephasing channel on all three squares
    for q in range(3):
        rho = sum(reduce(np.kron, [K if r == q else np.eye(2) for r in range(3)]) @ rho
                  @ reduce(np.kron, [K if r == q else np.eye(2) for r in range(3)]).conj().T for K in kraus("dephasing", p))
    return rho
def negativity(rho):   # targets (a, b): trace out s, partial transpose on b
    r = rho.reshape(2, 4, 2, 4).trace(axis1=0, axis2=2).reshape(2, 2, 2, 2)
    return -sum(e for e in np.linalg.eigvalsh(r.transpose(0, 3, 2, 1).reshape(4, 4)) if e < 0)

coh = np.outer(psi, psi.conj())
inc = (np.outer(ket("001"), ket("001")) + np.outer(ket("010"), ket("010"))) / 2   # same square probabilities, no phase
rec = lambda rho: np.real(ket("100") @ MERGE @ rho @ MERGE.conj().T @ ket("100"))
for p in np.linspace(0, 1, 101):
    k = (1 - p) ** 2
    Pq, Pc = rec(noisy(coh, p)), rec(noisy(inc, p))
    rq = (1 + k) / 2 * np.outer(psi, psi.conj()) + (1 - k) / 2 * np.outer(psi_perp, psi_perp.conj())
    assert np.allclose(noisy(coh, p), rq)   # decomposition of the dephased split state
    assert np.isclose(Pq, (1 + k) / 2) and np.isclose(Pc, 0.5) and np.isclose(negativity(noisy(coh, p)), k / 2)
for p in (0, 0.3, 0.6):
    k = (1 - p) ** 2
    print(f"p={p}: coherent recovery {(1 + k) / 2:.3f}, incoherent {0.5:.3f}, margin {k / 2:.3f}, target negativity {k / 2:.3f}")
print("chess checks passed")
