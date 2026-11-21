import numpy as np
from qgames import J, PA, PB, X, Y, Z, I2, payoffs_A

G = np.pi / 2
JG = J(G); JD = JG.conj().T

R = [(0.25, I2), (0.25, 1j * X), (0.25, 1j * Y), (0.25, 1j * Z)]
EW_A = [(0.5, I2), (0.5, -1j * Z)]
EW_B = [(0.5, np.array([[0, 1], [-1, 0]], dtype=complex)), (0.5, np.array([[0, -1j], [-1j, 0]]))]

def final_rho(UA, UB):
    """|00> -> J -> UA x UB -> J^dag; returns final density matrix."""
    psi = JD @ np.kron(UA, UB) @ JG[:, 0]
    return np.outer(psi, psi.conj())

def pair_payoff(UA, UB):
    pr = np.real(np.diag(final_rho(UA, UB)))
    return np.array([pr @ PA, pr @ PB])

def mix_payoff(SA, SB):
    return sum(pa * pb * pair_payoff(UA, UB) for pa, UA in SA for pb, UB in SB)

# 1. (R,R) payoff and averaged final state
print("(R,R) payoffs (A, B):", mix_payoff(R, R))
rho_avg = sum(pa * pb * final_rho(UA, UB) for pa, UA in R for pb, UB in R)
print("averaged final rho equals I/4:", np.allclose(rho_avg, np.eye(4) / 4))
print("trace of averaged rho:", np.real(np.trace(rho_avg)))

# 2. NE check against Haar-random SU(2)
rng = np.random.default_rng(0)
v = rng.normal(size=(20000, 4)); v /= np.linalg.norm(v, axis=1, keepdims=True)
HAAR = [np.array([[a + 1j * b, c + 1j * d], [-c + 1j * d, a - 1j * b]]) for a, b, c, d in v]

dev_A = np.array([sum(p * pair_payoff(UA, UB)[0] for p, UB in R) for UA in HAAR])
ref_A = sum(p * payoffs_A(np.array(HAAR), UB, G) for p, UB in R)
assert np.allclose(dev_A, ref_A), "mismatch with qgames.payoffs_A"
dev_B = np.array([sum(p * pair_payoff(UA, UB)[1] for p, UA in R) for UB in HAAR])
print(f"Alice deviating vs R: max {dev_A.max():.10f}, min {dev_A.min():.10f}")
print(f"Bob deviating vs R:   max {dev_B.max():.10f}, min {dev_B.min():.10f}")
print("(R,R) weak NE (all deviations <= 2.25 + 1e-9):", max(dev_A.max(), dev_B.max()) <= 2.25 + 1e-9)

# 3. Eisert-Wilkens mixed NE
print("EW payoffs (A, B):", mix_payoff(EW_A, EW_B))
ew_dev_A = np.array([sum(p * pair_payoff(UA, UB)[0] for p, UB in EW_B) for UA in HAAR])
ew_dev_B = np.array([sum(p * pair_payoff(UA, UB)[1] for p, UA in EW_A) for UB in HAAR])
print(f"EW max deviation payoff, Alice: {ew_dev_A.max():.10f}")
print(f"EW max deviation payoff, Bob:   {ew_dev_B.max():.10f}")
