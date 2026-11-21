"""Quantum games under noise: CHSH vs EWL Prisoner's Dilemma.

Exact density-matrix simulation in numpy (fast; use for grids/Nash search).
Noise channels follow the course notes (Topic 9):
  depolarizing  eq 9.1/9.2 : rho -> (1-p) rho + p/3 sum_i s_i rho s_i   (Bloch shrink 1-4p/3)
  dephasing     eq 9.8/9.9 : rho -> (1-p/2) rho + p/2 Z rho Z
  amp. damping  eq 9.15    : E0 = diag(1, sqrt(1-p)), E1 = [[0, sqrt p],[0,0]]
The same channel acts independently on each qubit (keeps both games symmetric).

Run:  python qgames.py      (asserts + figures into ./figures)
"""
import os
import numpy as np
from scipy.optimize import brentq

I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1.])

# ---------- channels (course conventions) ----------
def kraus(kind, p):
    if kind == "depolarizing":
        return [np.sqrt(1 - p) * I2] + [np.sqrt(p / 3) * s for s in (X, Y, Z)]
    if kind == "dephasing":
        return [np.sqrt(1 - p / 2) * I2, np.sqrt(p / 2) * Z]
    if kind == "amplitude":
        return [np.diag([1, np.sqrt(1 - p)]), np.array([[0, np.sqrt(p)], [0, 0]])]
    raise ValueError(kind)

def local_noise(rho, kind, p):
    """Same single-qubit channel on both qubits: sum_jk (Ej x Ek) rho (Ej x Ek)^dag."""
    K = kraus(kind, p)
    return sum(np.kron(a, b) @ rho @ np.kron(a, b).conj().T for a in K for b in K)

# ---------- CHSH ----------
PHI = np.array([1, 0, 0, 1]) / np.sqrt(2)

def obs(angle):  # +-1 observable cos(a) Z + sin(a) X
    return np.cos(angle) * Z + np.sin(angle) * X

def chsh_S(rho, a=(0, np.pi / 2), b=(np.pi / 4, -np.pi / 4)):
    E = lambda x, y: np.real(np.trace(rho @ np.kron(obs(x), obs(y))))
    return E(a[0], b[0]) + E(a[0], b[1]) + E(a[1], b[0]) - E(a[1], b[1])

def chsh(kind="depolarizing", p=0.0):
    rho = local_noise(np.outer(PHI, PHI.conj()), kind, p)
    S = chsh_S(rho)
    return S, 0.5 + S / 8  # P_win = 1/2 + S/8 (uniform inputs, +-1 outcomes)

PAULIS = (X, Y, Z)

def chsh_optimal(kind="depolarizing", p=0.0):
    """Max over all settings of the CHSH value (Horodecki criterion, PLA 200, 340 (1995)):
    S_max = 2 sqrt(sum of the two largest eigenvalues of T^T T), T_ij = tr(rho s_i x s_j)."""
    rho = local_noise(np.outer(PHI, PHI.conj()), kind, p)
    T = np.array([[np.real(np.trace(rho @ np.kron(a, b))) for b in PAULIS] for a in PAULIS])
    return 2 * np.sqrt(np.sort(np.linalg.eigvalsh(T.T @ T))[-2:].sum())

def chsh_classical_max():
    """Brute force all 16 deterministic strategies a(x), b(y); win iff a XOR b = x AND y."""
    best = 0
    for fa in range(4):
        for fb in range(4):
            wins = sum(((fa >> x) & 1) ^ ((fb >> y) & 1) == (x & y) for x in (0, 1) for y in (0, 1))
            best = max(best, wins / 4)
    return best

# ---------- EWL Prisoner's Dilemma ----------
def U(t, f):
    return np.array([[np.exp(1j * f) * np.cos(t / 2), np.sin(t / 2)],
                     [-np.sin(t / 2), np.exp(-1j * f) * np.cos(t / 2)]])

C, D, Q = U(0, 0), U(np.pi, 0), U(0, np.pi / 2)
PA = np.array([3, 0, 5, 1]); PB = np.array([3, 5, 0, 1])  # order CC, CD, DC, DD (Alice first)

def J(g):  # exp(i g D x D / 2); D x D squares to identity
    return np.cos(g / 2) * np.eye(4) + 1j * np.sin(g / 2) * np.kron(D, D)

def ewl(UA, UB, g, kind="depolarizing", p=0.0):
    """|00> -> J -> UA x UB -> noise -> J^dag -> measure. Returns (payoff A, payoff B)."""
    psi = np.kron(UA, UB) @ J(g)[:, 0]
    rho = local_noise(np.outer(psi, psi.conj()), kind, p)
    pr = np.real(np.diag(J(g).conj().T @ rho @ J(g)))
    return pr @ PA, pr @ PB

GRID = np.array([U(t, f) for t in np.linspace(0, np.pi, 41) for f in np.linspace(0, np.pi / 2, 21)])  # (861,2,2)

def payoffs_A(UAs, UB, g, kind="depolarizing", p=0.0):
    """Batched ewl(UA, UB, g, kind, p)[0] over a stack UAs of shape (N,2,2).
    Noise is a 16x16 superoperator on vec(rho) and the measurement + payoff is a 16-vector,
    so the whole batch is one matmul: payoff = vec(rho) . (Sup^T c)."""
    Jg, Jd = J(g), J(g).conj().T
    Ms = [np.kron(a, b) for a in kraus(kind, p) for b in kraus(kind, p)]
    Sup = sum(np.kron(M, M.conj()) for M in Ms)                          # vec(M rho M^dag) = kron(M, M*) vec(rho)
    c = np.einsum('i,ia,bi->ab', PA, Jd, Jg).reshape(16)                 # payoff = sum_ab c[a,b] rho_out[a,b]
    v = Sup.T @ c
    psi = np.einsum('nab,cd->nacbd', UAs, UB).reshape(-1, 4, 4) @ Jg[:, 0]  # kron(UA,UB) J|00>, (N,4)
    vec_rho = (psi[:, :, None] * psi.conj()[:, None, :]).reshape(-1, 16)
    return np.real(vec_rho @ v)

def is_nash(S, g, kind="depolarizing", p=0.0, tol=1e-9):
    """Symmetric profile (S,S) in the 2-parameter EWL set. Game + noise are symmetric,
    so checking Alice's deviations is enough. Uses a fixed 41x21 deviation grid."""
    base = ewl(S, S, g, kind, p)[0]
    return payoffs_A(GRID, S, g, kind, p).max() <= base + tol

# ---------- self-checks ----------
def checks():
    assert chsh_classical_max() == 0.75
    S, P = chsh()
    assert np.isclose(S, 2 * np.sqrt(2)) and np.isclose(P, np.cos(np.pi / 8) ** 2)
    # local depolarizing shrinks each Bloch vector by eta = 1 - 4p/3, correlators by eta^2
    p = 0.1; eta = 1 - 4 * p / 3
    assert np.isclose(chsh("depolarizing", p)[0], 2 * np.sqrt(2) * eta ** 2)
    # optimal settings: depolarizing is Werner-type, so fixed settings are already optimal
    assert np.isclose(chsh_optimal("depolarizing", p), chsh("depolarizing", p)[0])
    for p in (0.1, 0.25):
        assert np.isclose(chsh_optimal("amplitude", p), 2 * np.sqrt(2) * (1 - p))
    assert np.isclose(chsh_optimal("amplitude", 1 - 1 / np.sqrt(2)), 2)   # p* = 1 - 1/sqrt2
    assert np.isclose(chsh_optimal("dephasing", 0.9), 2 * np.sqrt(1 + 0.1 ** 4)) and chsh_optimal("dephasing", 0.9) > 2
    g = np.pi / 2
    assert np.allclose(ewl(Q, Q, g), (3, 3)) and np.allclose(ewl(D, D, g), (1, 1))
    assert np.allclose(ewl(C, D, 0), (0, 5)) and np.allclose(ewl(D, C, 0), (5, 0))
    assert is_nash(D, 0.40) and not is_nash(D, 0.50)          # DD NE up to arcsin(1/sqrt5)=0.4636
    assert is_nash(Q, 0.72) and not is_nash(Q, 0.65)          # QQ NE from arcsin(sqrt(2/5))=0.6847
    print("all checks passed")

META = {"CreationDate": None}  # byte-stable PDFs on rerun

# ---------- figures ----------
def figures(out="figures"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    ps = np.linspace(0, 0.75, 31)
    kinds = ["depolarizing", "dephasing", "amplitude"]

    # 1. CHSH vs noise: optimised settings (solid) and fixed noiseless settings (dashed)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for i, k in enumerate(kinds):
        ax.plot(ps, [chsh_optimal(k, p) for p in ps], c=f"C{i}", label=k)
        ax.plot(ps, [chsh(k, p)[0] for p in ps], c=f"C{i}", ls="--")
    ax.axhline(2, ls=":", c="k", lw=1, label="classical bound S=2")
    ax.set(xlabel="noise p (per qubit)", ylabel="CHSH value S", title="CHSH under local noise")
    ax.legend(fontsize=8, title="solid: optimised settings\ndashed: fixed noiseless settings", title_fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/chsh_noise.pdf", metadata=META); plt.close(fig)

    # 2. EWL vs gamma (noiseless): each profile's payoff vs Alice's best unilateral deviation.
    #    A profile is a NE exactly where its payoff >= best deviation.
    gs = np.linspace(0, np.pi / 2, 161)
    G = np.array(GRID)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for S, name, c in ((Q, "Q", "C0"), (D, "D", "C1")):
        ax.plot(gs, [ewl(S, S, g)[0] for g in gs], c=c, label=f"({name},{name}) payoff")
        ax.plot(gs, [payoffs_A(G, S, g).max() for g in gs], c=c, ls="--", label=f"best deviation vs {name}")
    for v in (np.arcsin(1 / np.sqrt(5)), np.arcsin(np.sqrt(2 / 5))):
        ax.axvline(v, ls=":", c="gray")
    ax.set(xlabel=r"entanglement $\gamma$", ylabel="Alice payoff",
           title=r"EWL: NE where payoff = best deviation")
    ax.text(0.12, 2.2, "(D,D) NE", fontsize=8); ax.text(1.1, 2.2, "(Q,Q) NE", fontsize=8)
    ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout(); fig.savefig(f"{out}/ewl_gamma.pdf", metadata=META); plt.close(fig)

    # 3. Phase diagram: is (Q,Q) a Nash equilibrium over (gamma, p)? + its payoff (one shared colour scale)
    gs2, ps2 = np.linspace(0, np.pi / 2, 41), np.linspace(0, 0.6, 41)
    titles = {"depolarizing": "Depolarizing", "dephasing": "Dephasing", "amplitude": "Amplitude damping"}
    data = {k: (np.array([[is_nash(Q, g, k, p) for g in gs2] for p in ps2]),
                np.array([[ewl(Q, Q, g, k, p)[0] for g in gs2] for p in ps2])) for k in kinds}
    vmin = min(d[1].min() for d in data.values()); vmax = max(d[1].max() for d in data.values())
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.6), sharey=True, layout="constrained")
    for ax, k in zip(axs, kinds):
        im = ax.pcolormesh(gs2, ps2, data[k][1], shading="auto", cmap="viridis", vmin=vmin, vmax=vmax, rasterized=True)
        ax.contour(gs2, ps2, data[k][0].astype(float), levels=[0.5], colors="w", linewidths=2)
        ax.set_title(titles[k].replace("Amplitude damping", "Ampl. damping"), fontsize=12); ax.set_xlabel(r"entanglement $\gamma$", fontsize=12)
        ax.tick_params(labelsize=11); ax.set_xticks([0, 0.5, 1.0, 1.5])
    axs[0].set_ylabel("noise p", fontsize=12)
    cb = fig.colorbar(im, ax=axs, pad=0.02); cb.set_label("(Q,Q) payoff", fontsize=12); cb.ax.tick_params(labelsize=11)
    fig.savefig(f"{out}/ewl_phase.pdf", metadata=META); plt.close(fig)
    for k in kinds:  # drop the superseded per-channel files
        f = f"{out}/ewl_phase_{k}.pdf"
        if os.path.exists(f): os.remove(f)

    # 4. Matched comparison: quantum margin vs noise (CHSH: S_max-2; EWL at gamma=pi/2: $(Q,Q)-$(D,D))
    pn = np.linspace(0, 0.6, 61)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, k in enumerate(kinds):
        a1.plot(pn, [chsh_optimal(k, p) - 2 for p in pn], c=f"C{i}", label=titles[k])
        a2.plot(pn, [ewl(Q, Q, np.pi / 2, k, p)[0] - ewl(D, D, np.pi / 2, k, p)[0] for p in pn], c=f"C{i}", label=titles[k])
    for ax in (a1, a2): ax.axhline(0, c="k", lw=1)
    a1.set(xlabel="noise p (per qubit)", ylabel=r"$S_{\max}-2$", title="CHSH quantum margin (>0: violation)")
    a2.set(xlabel="noise p (per qubit)", ylabel="(Q,Q) payoff minus (D,D) payoff", title=r"EWL quantum margin at $\gamma=\pi/2$")
    a1.legend(fontsize=9); fig.tight_layout(); fig.savefig(f"{out}/advantage_vs_noise.pdf", metadata=META); plt.close(fig)

    # 5. EWL protocol circuit schematic (matplotlib; qiskit's mpl drawer needs pylatexenc, which is not in requirements)
    fig, ax = plt.subplots(figsize=(8, 2.4)); ax.axis("off"); ax.set(xlim=(0, 10.2), ylim=(-0.2, 2.2))
    for y, nm in ((1.6, "A"), (0.4, "B")):
        ax.plot([0.5, 9.8], [y, y], c="k", lw=1, zorder=0)
        ax.text(0.4, y, r"$|0\rangle$", ha="right", va="center", fontsize=12)
    box = lambda x, y, t, w=0.9, h=0.7, c="w": (ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, fc=c, ec="k", zorder=2)),
                                                 ax.text(x, y, t, ha="center", va="center", fontsize=12, zorder=3))
    def two(x, t):  # two-qubit gate spanning both wires
        ax.add_patch(plt.Rectangle((x - 0.45, 0.05), 0.9, 1.9, fc="#dde6f5", ec="k", zorder=2)); ax.text(x, 1.0, t, ha="center", va="center", fontsize=12, zorder=3)
    two(1.5, r"$J(\gamma)$")
    box(3.3, 1.6, r"$U_A$"); box(3.3, 0.4, r"$U_B$")
    for y in (1.6, 0.4): box(5.1, y, r"$\mathcal{E}_p$", c="#f7dcdc")
    two(6.9, r"$J^\dagger$")
    for y in (1.6, 0.4): box(8.7, y, "meas.", w=1.0)
    ax.text(5.1, 2.1, "noise", ha="center", fontsize=10, color="#a33")
    ax.text(3.3, 2.1, "moves", ha="center", fontsize=10); ax.text(1.5, 2.1, "arbiter", ha="center", fontsize=10); ax.text(6.9, 2.1, "arbiter", ha="center", fontsize=10)
    fig.savefig(f"{out}/ewl_circuit.pdf", metadata=META, bbox_inches="tight"); plt.close(fig)
    print(f"figures written to {os.path.abspath(out)}")

def thresholds():
    """Noise/entanglement thresholds quoted in the report."""
    kinds = ["depolarizing", "dephasing", "amplitude"]
    def gamma_star(k, p):  # bisection on is_nash, bracket [0, pi/2], tol 1e-4
        lo, hi = 0.0, np.pi / 2
        assert not is_nash(Q, lo, k, p) and is_nash(Q, hi, k, p)
        while hi - lo > 1e-4:
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if is_nash(Q, mid, k, p) else (mid, hi)
        return (lo + hi) / 2
    lo, hi = 0.0, np.pi / 2   # noiseless (D,D): NE for small gamma, bisect where it stops being one
    while hi - lo > 1e-4:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if is_nash(D, mid) else (lo, mid)
    print(f"noiseless thresholds: (D,D) NE up to gamma={(lo + hi) / 2:.4f} (arcsin(1/sqrt5)={np.arcsin(1 / np.sqrt(5)):.4f}); "
          f"(Q,Q) NE from gamma={gamma_star('depolarizing', 0.0):.4f} (arcsin(sqrt(2/5))={np.arcsin(np.sqrt(0.4)):.4f})")
    print("EWL (Q,Q) NE threshold gamma*, and payoffs at gamma=pi/2")
    for k in kinds:
        for p in (0.1, 0.3):
            print(f"  {k:13s} p={p}: gamma*={gamma_star(k, p):.3f}  $(Q,Q)={ewl(Q, Q, np.pi / 2, k, p)[0]:.3f}  "
                  f"$(D,D)={ewl(D, D, np.pi / 2, k, p)[0]:.3f}")
    print("margins at p=0.6: " + "; ".join(f"{k}: CHSH S_max-2={chsh_optimal(k, 0.6) - 2:.3f}, EWL $(Q,Q)-$(D,D)="
          f"{ewl(Q, Q, np.pi / 2, k, 0.6)[0] - ewl(D, D, np.pi / 2, k, 0.6)[0]:.3f}, (Q,Q) NE at pi/2: {is_nash(Q, np.pi / 2, k, 0.6)}" for k in kinds))
    print("CHSH p* (S=2): fixed settings / optimised settings")
    for k in kinds:
        pf = brentq(lambda p: chsh(k, p)[0] - 2, 1e-6, 0.99)
        po = brentq(lambda p: chsh_optimal(k, p) - 2, 1e-6, 0.99) if chsh_optimal(k, 0.99) < 2 else float("inf")
        print(f"  {k:13s} fixed {pf:.4f}  optimised " + (f"{po:.4f}" if po != float("inf") else "none (S_max>2 for all p<1)"))
    print(f"  AD fixed closed form (3-sqrt(8 sqrt2-7))/4 = {(3 - np.sqrt(8 * np.sqrt(2) - 7)) / 4:.4f}; 1-1/sqrt2 = {1 - 1 / np.sqrt(2):.4f}")
    print(f"  dephasing optimised S_max at p=0.9: {chsh_optimal('dephasing', 0.9):.4f}, at p=0.99: {chsh_optimal('dephasing', 0.99):.4f}")

if __name__ == "__main__":
    checks()
    thresholds()
    figures(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
