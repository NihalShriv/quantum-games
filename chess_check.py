"""Quantum chess split/merge (Cantwell, arXiv:1906.05836) on three squares s, a, b under the three course channels.
Qubit = occupancy of a square. W = iSWAP, V = sqrt(iSWAP). Split s -> (a, b): W_sb V_sa. Merge: V_sa W_sb.
Noise acts on the two target squares between split and merge (the empty source is left noiseless).
Checks, for every channel and 101 values of p:
  recovery = <psi|rho|psi> (lemma), coherent and classical recovery, target negativity, against closed forms;
  a noiseless Qiskit cross-check of split and merge; writes figures/chess_noise.pdf.
Run: python chess_check.py (a few seconds)."""
import os
import numpy as np
from functools import reduce
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qgames import kraus, META

h = 1 / np.sqrt(2)
W = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
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
psi_perp = 1j * (ket("001") - ket("010")) / np.sqrt(2)
assert np.allclose(psi, 1j * (ket("001") + ket("010")) / np.sqrt(2))
assert np.allclose(MERGE @ psi, -ket("100"))                       # merge undoes the split, phase -1
assert np.allclose(MERGE @ psi_perp, -1j * ket("010"))             # orthogonal branch state never returns

def noisy(rho, kind, p):   # same channel on the two target squares a, b
    for q in (1, 2):
        rho = sum(reduce(np.kron, [K if r == q else np.eye(2) for r in range(3)]) @ rho
                  @ reduce(np.kron, [K if r == q else np.eye(2) for r in range(3)]).conj().T for K in kraus(kind, p))
    return rho
def negativity(rho):   # targets (a, b): trace out s, partial transpose on b
    r = rho.reshape(2, 4, 2, 4).trace(axis1=0, axis2=2).reshape(2, 2, 2, 2)
    return -sum(e for e in np.linalg.eigvalsh(r.transpose(0, 3, 2, 1).reshape(4, 4)) if e < 0)

coh = np.outer(psi, psi.conj())
inc = (np.outer(ket("001"), ket("001")) + np.outer(ket("010"), ket("010"))) / 2   # same square probabilities, no phase
assert np.allclose(inc, (coh + np.outer(psi_perp, psi_perp.conj())) / 2)
rec = lambda rho: np.real(ket("100") @ MERGE @ rho @ MERGE.conj().T @ ket("100"))

def closed(kind, p):   # (coherent recovery, classical recovery, negativity)
    if kind == "dephasing":
        k = (1 - p) ** 2
        return (1 + k) / 2, 0.5, k / 2
    if kind == "depolarizing":
        v = (1 - 4 * p / 3) ** 2
        return (1 + 3 * v) / 4, (1 + v) / 4, max(0.0, (3 * v - 1) / 4)
    return 1 - p, (1 - p) / 2, (np.sqrt(p ** 2 + (1 - p) ** 2) - p) / 2

KINDS = ["depolarizing", "dephasing", "amplitude"]
rng = np.random.default_rng(0)
for kind in KINDS:
    for p in np.linspace(0, 1, 101):
        rq, rc = noisy(coh, kind, p), noisy(inc, kind, p)
        A = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)); R = A @ A.conj().T
        R = R / np.trace(R)   # lemma on a random state: recovery = <psi|rho|psi>
        assert np.isclose(rec(R), np.real(psi.conj() @ R @ psi))
        assert np.allclose([rec(rq), rec(rc), negativity(rq)], closed(kind, p)), (kind, p)
        assert np.isclose(np.trace(rq).real, 1) and np.isclose(np.trace(rc).real, 1)

# separability point and zero of the margin under depolarizing noise
p_sep = 0.75 * (1 - 1 / np.sqrt(3))
assert closed("depolarizing", p_sep - 1e-6)[2] > 0 and np.isclose(closed("depolarizing", p_sep + 1e-6)[2], 0)
for kind in KINDS:
    for p in (0.1, 0.3, 0.6):
        q, c, n = closed(kind, p)
        print(f"{kind:13s} p={p}: coherent {q:.3f}  classical {c:.3f}  margin {q - c:.3f}  negativity {n:.4f}")
assert all(closed("depolarizing", x)[0] > closed("depolarizing", x)[1] for x in np.linspace(0, 1, 1001) if abs(x - 0.75) > 1e-9)
print(f"depolarizing: targets separable from p = 3/4 (1 - 1/sqrt3) = {p_sep:.4f}; margin v/2 > 0 for every p != 3/4")

# Qiskit cross-check (noiseless): same split and merge with Qiskit's iSwapGate and its square root
from qiskit import QuantumCircuit
from qiskit.circuit.library import iSwapGate
from qiskit.quantum_info import Statevector
qc = QuantumCircuit(3)                  # qubit 0 = s, 1 = a, 2 = b
qc.x(0)
qc.append(iSwapGate().power(0.5), [0, 1]); qc.append(iSwapGate(), [0, 2])          # split
sv = Statevector(qc).probabilities_dict()
assert set(sv) == {"010", "100"} and all(np.isclose(x, 0.5) for x in sv.values())   # little-endian: a or b occupied
qc.append(iSwapGate(), [0, 2]); qc.append(iSwapGate().power(0.5), [0, 1])          # merge
assert np.isclose(Statevector(qc).probabilities_dict().get("001", 0), 1)          # piece back on s
print("Qiskit split/merge agrees")

# figure
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(out, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
cols = {"depolarizing": "C0", "dephasing": "C1", "amplitude": "C2"}
for kind in KINDS:
    ps = np.linspace(0, 1, 201)
    v = np.array([closed(kind, p) for p in ps])
    ax[0].plot(ps, v[:, 0], color=cols[kind], label=kind if kind != "amplitude" else "amplitude damping")
    ax[0].plot(ps, v[:, 1], color=cols[kind], ls="--")
ax[0].set(xlabel="noise p (per square)", ylabel="probability piece returns to s", title="Split then merge: coherent (solid) vs classical (dashed)", xlim=(0, 1), ylim=(0, 1.02))
ax[0].legend(fontsize=9)
ps = np.linspace(0, 1, 401); vv = (1 - 4 * ps / 3) ** 2
ax[1].plot(ps, vv / 2, color="C0", label="recovery margin v/2 (coherence)")
ax[1].plot(ps, np.maximum(0, (3 * vv - 1) / 4), color="C0", ls="--", label="negativity (entanglement)")
ax[1].plot(ps, np.maximum(0, 2 * np.sqrt(2) * vv - 2) / 4, color="C0", ls=":", label=r"$(S-2)/4$ for $\Phi^+$ (CHSH violation)")
for x, lab in ((0.75 * (1 - 2 ** -0.25), "0.119"), (p_sep, "0.317"), (0.75, "0.75")):
    ax[1].axvline(x, color="gray", lw=0.6); ax[1].text(x, 0.47, lab, ha="center", fontsize=9, backgroundcolor="w")
ax[1].set(xlabel="depolarizing noise p (per qubit)", title=r"Depolarizing: $v\rho+(1-v)I/4$, $v=(1-4p/3)^2$", xlim=(0, 1), ylim=(0, 0.52))
ax[1].legend(fontsize=9, loc="center right")
fig.tight_layout(); fig.savefig(f"{out}/chess_noise.pdf", metadata=META); plt.close(fig)

# figure: board picture and circuit of split -> noise -> merge
fig, (axb, axc) = plt.subplots(1, 2, figsize=(10, 2.8), gridspec_kw={"width_ratios": [1, 2.6]})
axb.axis("off"); axb.set(xlim=(-0.2, 3.2), ylim=(-0.3, 2.4), aspect="equal")
for i in range(3):
    for j in range(2):
        axb.add_patch(plt.Rectangle((i, j), 1, 1, fc="#e8d9b5" if (i + j) % 2 else "#b58863", ec="k", lw=0.5))
axb.text(1.5, 0.5, "N", ha="center", va="center", fontsize=18, fontweight="bold")
axb.text(1.5, 0.12, "s", ha="center", fontsize=9)
for x, lab in ((0.5, "a"), (2.5, "b")):
    axb.text(x, 1.5, "N", ha="center", va="center", fontsize=18, color="gray", alpha=0.6)
    axb.text(x, 1.12, lab, ha="center", fontsize=9)
    axb.annotate("", xy=(x + (0.2 if x < 1 else -0.2), 1.25), xytext=(1.5 + (-0.25 if x < 1 else 0.25), 0.8),
                 arrowprops=dict(arrowstyle="->", color="C0"))
axb.text(1.5, 2.15, r"split: amplitude $1/\sqrt{2}$ on $a$ and on $b$", ha="center", fontsize=9)
axc.axis("off"); axc.set(xlim=(0, 11), ylim=(-0.3, 2.6))
Y = {"s": 2.0, "a": 1.0, "b": 0.0}
for nm, y in Y.items():
    axc.plot([0.8, 10.4], [y, y], c="k", lw=1, zorder=0)
    axc.text(0.7, y, f"{nm}: " + (r"$|1\rangle$" if nm == "s" else r"$|0\rangle$"), ha="right", va="center", fontsize=11)
def gate(x, q1, q2, t):   # two-qubit gate on wires q1, q2
    axc.plot([x, x], [Y[q1], Y[q2]], c="k", lw=1, zorder=1)
    for q in (q1, q2):
        axc.add_patch(plt.Rectangle((x - 0.45, Y[q] - 0.3), 0.9, 0.6, fc="#dde6f5", ec="k", zorder=2))
        axc.text(x, Y[q], t, ha="center", va="center", fontsize=10, zorder=3)
gate(1.6, "s", "a", r"$V$"); gate(2.9, "s", "b", r"$W$")
for q in ("a", "b"):
    axc.add_patch(plt.Rectangle((4.75, Y[q] - 0.3), 0.9, 0.6, fc="#f7dcdc", ec="k", zorder=2))
    axc.text(5.2, Y[q], r"$\mathcal{E}_p$", ha="center", va="center", fontsize=11, zorder=3)
gate(7.0, "s", "b", r"$W$"); gate(8.3, "s", "a", r"$V$")
for q in Y:
    axc.add_patch(plt.Rectangle((9.35, Y[q] - 0.3), 0.9, 0.6, fc="w", ec="k", zorder=2)); axc.text(9.8, Y[q], "meas.", ha="center", va="center", fontsize=9, zorder=3)
for x, t, c in ((2.25, "split", "k"), (5.2, "noise", "#a33"), (7.65, "merge", "k"), (9.8, r"recovered if $|100\rangle$", "k")):
    axc.text(x, 2.45, t, ha="center", fontsize=10, color=c)
fig.tight_layout(); fig.savefig(f"{out}/chess_circuit.pdf", metadata=META, bbox_inches="tight"); plt.close(fig)
print("chess checks passed; figures written to figures/chess_noise.pdf and figures/chess_circuit.pdf")
