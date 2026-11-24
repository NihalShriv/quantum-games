"""Exact results at gamma = pi/2 under local noise (SymPy), checked against the numpy model.
1. Dephasing: Alice's payoff against Q in closed form, and the exact threshold p = 1 - 1/sqrt7.
2. Classical C/D game inside the same noisy device: when (D,D) stops being its equilibrium."""
import numpy as np
import sympy as sp
from sympy.physics.quantum import TensorProduct as T
import qgames as qg

t, f, p = sp.symbols('theta phi p', positive=True)
k = (1 - p) ** 2
def U(t, f):
    return sp.Matrix([[sp.exp(sp.I * f) * sp.cos(t / 2), sp.sin(t / 2)], [-sp.sin(t / 2), sp.exp(-sp.I * f) * sp.cos(t / 2)]])
C, D, Q = sp.eye(2), U(sp.pi, 0), U(0, sp.pi / 2)
X, Y, Z, I2 = sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.diag(1, -1), sp.eye(2)
KRAUS = {"dephasing": [sp.sqrt(1 - p / 2) * I2, sp.sqrt(p / 2) * Z],
         "depolarizing": [sp.sqrt(1 - p) * I2] + [sp.sqrt(p / 3) * s for s in (X, Y, Z)],
         "amplitude": [sp.diag(1, sp.sqrt(1 - p)), sp.Matrix([[0, sp.sqrt(p)], [0, 0]])]}
J = (sp.eye(4) + sp.I * T(D, D)) / sp.sqrt(2)          # J(pi/2)

def pay(UA, UB, ch):
    psi = T(UA, UB) * J[:, 0]; r = psi * psi.H
    r = sum((T(a, b) * r * T(a, b).H for a in KRAUS[ch] for b in KRAUS[ch]), sp.zeros(4))
    out = J.H * r * J
    return sp.simplify(sp.expand(sum(c * out[i, i] for i, c in enumerate([3, 0, 5, 1]))))

# 1. dephasing, Alice's U(theta,phi) against Q
c2, s2f = sp.cos(t / 2) ** 2, sp.sin(f) ** 2
claim = sp.Rational(5, 2) * (1 - k) + c2 * (2 * k * s2f + (3 * k - 1) / 2)
def zero(e, n=40):   # identity check at random points; sqrt(2-p) with p in [0,1] defeats simplify
    rng = np.random.default_rng(0)
    for _ in range(n):
        v = {t: rng.uniform(0, np.pi), f: rng.uniform(0, np.pi / 2), p: rng.uniform(0, 1)}
        assert abs(complex(sp.N(e.subs(v)))) < 1e-12
    return True
A = pay(U(t, f), Q, "dephasing")
assert zero(A - claim) and zero(pay(Q, Q, "dephasing") - (2 + k)) and zero(pay(D, Q, "dephasing") - sp.Rational(5, 2) * (1 - k))
pstar = 1 - 1 / np.sqrt(7)
print(f"1. dephasing: $_A(U,Q) = 5(1-k)/2 + cos^2(theta/2)[2k sin^2 phi + (3k-1)/2], k=(1-p)^2; threshold p = 1-1/sqrt7 = {pstar:.6f}")
assert qg.is_nash(qg.Q, np.pi / 2, "dephasing", pstar - 1e-4) and not qg.is_nash(qg.Q, np.pi / 2, "dephasing", pstar + 1e-4)

# 2. classical C/D game in the noisy device
r = sp.Symbol('r', positive=True)
def clean(e):   # p is in [0,1], so sqrt(2-p) is real: remove the conjugates sympy keeps
    for x in (2 - p, 1 - p):
        e = sp.expand(e.subs(sp.sqrt(x), r).subs(sp.conjugate(r), r)).subs(r ** 2, x)
        e = e.subs(r, sp.sqrt(x))
    return sp.factor(sp.simplify(e))
print("2. classical C/D game at gamma=pi/2: a=$(C,C) b=$(C,D) c=$(D,C) d=$(D,D), Alice's payoffs")
for ch in KRAUS:
    a, b, c, d = (clean(pay(x, y, ch)) for x, y in [(C, C), (C, D), (D, C), (D, D)])
    print(f"  {ch:13s} a={a}, b={b}, c={c}, d={d}\n  {'':13s} c-a={sp.factor(c - a)}, d-b={sp.factor(d - b)}")
    for pv in np.linspace(0, 1, 101):   # symbolic vs numpy
        num = [qg.ewl(x, y, np.pi / 2, ch, pv)[0] for x, y in [(qg.C, qg.C), (qg.C, qg.D), (qg.D, qg.C), (qg.D, qg.D)]]
        assert np.allclose([float(e.subs(p, pv)) for e in (a, b, c, d)], num)
    assert sp.simplify(c - a - {"dephasing": (3 * k + 1) / 2, "depolarizing": 2 * (4 * p - 3) ** 2 / 9, "amplitude": (1 - p) * (4 - p) / 2}[ch]) == 0
    assert sp.simplify(d - b - {"dephasing": (3 * k - 1) / 2, "depolarizing": (4 * p - 3) ** 2 / 9, "amplitude": (1 - p) * (2 + p) / 2}[ch]) == 0
    if ch == "dephasing":
        A = {"a": a, "b": b, "c": c, "d": d}

# dephasing: the classical C/D game above p = 1-1/sqrt3
a, b, c, d = A["a"], A["b"], A["c"], A["d"]
assert sp.simplify(a - (2 + k)) == 0 and sp.simplify(a - b - c + d + 1) == 0 and sp.simplify(b + c - 5) == 0
q = sp.simplify((d - b) / (a - b - c + d))                      # P(C) making the opponent indifferent
assert sp.simplify(q - (1 - 3 * k) / 2) == 0
mixed = sp.simplify(q * a + (1 - q) * b)                         # payoff of the symmetric mixed NE
assert sp.simplify(q * c + (1 - q) * d - mixed) == 0             # C and D pay the same against q: indifference
assert sp.simplify(2 + k - mixed - (7 * k - 1) * (1 + 3 * k) / 4) == 0
ps = 1 - 1 / np.sqrt(3)
for pv in np.linspace(0, 1, 1001):
    av, bv, cv, dv = (float(e.subs(p, pv)) for e in (a, b, c, d))
    assert (dv >= bv - 1e-12) == (pv <= ps + 1e-12)               # (D,D) NE iff p <= 1-1/sqrt3
    if pv > ps + 1e-12:
        assert bv >= dv and cv >= av and 0 < float(q.subs(p, pv)) < 1   # (C,D) is a NE and q is a probability
print(f"  dephasing: (D,D) is a classical NE iff p <= 1-1/sqrt3 = {ps:.4f}; above it the NE are (C,D), (D,C) and a symmetric mixed one")
print(f"  q = (1-3k)/2, $(Q,Q) - $(mixed) = (7k-1)(1+3k)/4 >= 0 iff p <= 1-1/sqrt7")
v = {p: sp.Rational(3, 5)}
print(f"  dephasing p=0.6: a,b,c,d = {[float(e.subs(v)) for e in (a, b, c, d)]}, (Q,Q)={float((2 + k).subs(v)):.4f}, "
      f"asymmetric NE mean={float(((b + c) / 2).subs(v)):.4f}, q(C)={float(q.subs(v)):.4f}, mixed payoff={float(mixed.subs(v)):.4f}")
assert abs(float(mixed.subs(v)) - 2.1156) < 1e-12 and abs(float(q.subs(v)) - 0.26) < 1e-12
print("exact checks passed")
