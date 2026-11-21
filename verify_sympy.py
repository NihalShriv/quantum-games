"""Symbolic checks of the EWL formulas in report/main.tex (Appendix A and Sec. 4.3)."""
import sympy as sp
from sympy import Matrix, kronecker_product as kp, I, cos, sin

th, ph, g = sp.symbols("theta phi gamma", real=True)

def U(t, f):
    return Matrix([[sp.exp(I * f) * cos(t / 2), sin(t / 2)], [-sin(t / 2), sp.exp(-I * f) * cos(t / 2)]])

D, Q = U(sp.pi, 0), U(0, sp.pi / 2)
X = Matrix([[0, 1], [1, 0]]); Z = sp.diag(1, -1)
e00 = Matrix([1, 0, 0, 0])
PA, PB = [3, 0, 5, 1], [3, 5, 0, 1]   # outcome order CC, CD, DC, DD

def J(gam):
    return cos(gam / 2) * sp.eye(4) + I * sin(gam / 2) * kp(D, D)

def final(UA, UB, gam):
    return J(gam).H * kp(UA, UB) * J(gam) * e00

def probs(UA, UB, gam):
    return [sp.simplify(sp.expand(sp.Abs(sp.expand_complex(x)) ** 2)) for x in final(UA, UB, gam)]

def zero(e):
    """Symbolic zero test; if simplify cannot reduce a trig identity, fall back to 30 random numeric points."""
    if sp.simplify(sp.expand(e)) == 0:
        return True
    import random
    syms = sorted(e.free_symbols, key=str)
    for _ in range(30):
        v = e.subs({x: random.uniform(-3, 3) for x in syms})
        if abs(complex(sp.N(v))) > 1e-12:
            return False
    return True

c, s = cos(th / 2), sin(th / 2)
# eqs (3), (4): outcome probabilities and payoffs of a deviation U(theta, phi)
pD, pQ = probs(U(th, ph), D, g), probs(U(th, ph), Q, g)
tgt_D = [0, (1 - sin(g) ** 2 * sin(ph) ** 2) * c ** 2, sin(g) ** 2 * sin(ph) ** 2 * c ** 2, s ** 2]
tgt_Q = [(1 - sin(g) ** 2 * cos(ph) ** 2) * c ** 2, sin(g) ** 2 * s ** 2, cos(g) ** 2 * s ** 2,
         sin(g) ** 2 * cos(ph) ** 2 * c ** 2]
assert all(zero(a - b) for a, b in zip(pD, tgt_D)) and all(zero(a - b) for a, b in zip(pQ, tgt_Q))
assert zero(sum(pD) - 1) and zero(sum(pQ) - 1)
assert zero(sum(w * p for w, p in zip(PA, pD)) - (5 * sin(g) ** 2 * sin(ph) ** 2 * c ** 2 + s ** 2))
assert zero(sum(w * p for w, p in zip(PA, pQ)) - ((3 - 2 * sin(g) ** 2 * cos(ph) ** 2) * c ** 2 + 5 * cos(g) ** 2 * s ** 2))

# Sec. 4.3: general SU(2) deviation against the mixed NE at gamma = pi/2
a1, a2, b1, b2 = sp.symbols("a1 a2 b1 b2", real=True)
a, b = a1 + I * a2, b1 + I * b2
W = Matrix([[a, b], [-sp.conjugate(b), sp.conjugate(a)]])
h = sp.pi / 2

def pay(UA, UB, w):
    return sum(x * p for x, p in zip(w, probs(UA, UB, h)))

A_D, A_X = pay(W, D, PA), pay(W, -I * X, PA)          # Alice deviates, Bob plays D or -iX
assert zero(A_D - (5 * a2 ** 2 + b1 ** 2 + 3 * b2 ** 2)) and zero(A_X - (5 * a1 ** 2 + 3 * b1 ** 2 + b2 ** 2))
assert zero((A_D + A_X) / 2 - (sp.Rational(5, 2) * (a1 ** 2 + a2 ** 2) + 2 * (b1 ** 2 + b2 ** 2)))
B_I, B_Z = pay(sp.eye(2), W, PB), pay(-I * Z, W, PB)  # Bob deviates, Alice plays I or -iZ
assert zero(B_I - (3 * a1 ** 2 + a2 ** 2 + 5 * b1 ** 2)) and zero(B_Z - (3 * a2 ** 2 + a1 ** 2 + 5 * b2 ** 2))
assert zero((B_I + B_Z) / 2 - (2 * (a1 ** 2 + a2 ** 2) + sp.Rational(5, 2) * (b1 ** 2 + b2 ** 2)))
assert [pay(u, v, PA) for u in (sp.eye(2), -I * Z) for v in (D, -I * X)] == [0, 5, 5, 0]

# Benjamin-Hayden: U_A = M conj(U_B) P^dagger gives |DC> for a general U(2) element U_B
al, be, de = sp.symbols("alpha beta delta", real=True)
UB = sp.exp(I * de) * U(2 * be, al)
M = Matrix([[0, -I], [1, 0]]); P = sp.diag(1, I)
psi = final(M * UB.conjugate() * P.H, UB, h).applyfunc(sp.expand_complex)
assert zero(sp.Abs(psi[2]) ** 2 - 1) and all(zero(psi[k]) for k in (0, 1, 3))
print("sympy checks passed")
