"""Mixed NE of the gamma=pi/2 EWL Prisoner's Dilemma over full SU(2). Run: python mixed_ne.py"""
import numpy as np
from qgames import J, ewl, PA, PB

g = np.pi / 2
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1.])
MIX_A = [I2, 1j * Z]; MIX_B = [1j * X, 1j * Y]          # each with prob 1/2

def pay(UA, UB):
    return ewl(UA, UB, g)

def mix_pay(As, Bs):
    r = np.mean([pay(a, b) for a in As for b in Bs], axis=0)
    return r

def haar(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    return np.array([[q[0] + 1j * q[1], q[2] + 1j * q[3]], [-q[2] + 1j * q[3], q[0] - 1j * q[1]]])

def best_dev(player, other, n, rng):
    best = -1
    for _ in range(n):
        V = haar(rng)
        v = np.mean([pay(V, b)[0] if player == 0 else pay(a, V)[1] for b in other] if player == 0
                    else [pay(a, V)[1] for a in other])
        best = max(best, v)
    return best

if __name__ == "__main__":
    # sanity: J is unitary; outcome distributions of the profile
    assert np.allclose(J(g) @ J(g).conj().T, np.eye(4))
    for a in MIX_A:
        for b in MIX_B:
            psi = np.kron(a, b) @ J(g)[:, 0]
            pr = np.real(np.diag(J(g).conj().T @ np.outer(psi, psi.conj()) @ J(g)))
            print("outcome probs CC,CD,DC,DD:", np.round(pr, 6))
            assert np.isclose(pr[0], 0) and np.isclose(pr[3], 0)   # only CD / DC
    base = mix_pay(MIX_A, MIX_B)
    print("mix payoff (A,B):", base)
    assert np.allclose(base, (2.5, 2.5))
    rng = np.random.default_rng(0)
    N = 20000
    bA = best_dev(0, MIX_B, N, rng); bB = best_dev(1, MIX_A, N, rng)
    print(f"best Alice deviation vs B-mix over {N} Haar unitaries: {bA:.6f}")
    print(f"best Bob   deviation vs A-mix over {N} Haar unitaries: {bB:.6f}")
    assert bA <= 2.5 + 1e-9 and bB <= 2.5 + 1e-9
    # Alice's deviation payoff is exactly 5/2 for many V (indifference), check a pure-strategy comparison
    print("max pure-strategy QQ deviation check, Q vs Q:", ewl(1j * Z * 0 + np.array([[1j, 0], [0, -1j]]), np.array([[1j, 0], [0, -1j]]), g))
    print("NE verified (numerically, up to random sampling)")
