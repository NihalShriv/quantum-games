"""Backs the convergence and threshold claims in the report (under a minute):
1. gamma* at p=0.1, 0.3 on 41x21, 121x61 and 401x201 deviation grids;
2. a single non-NE -> NE switch along a 400-point gamma scan (p=0.1, 0.3, 0.6, all channels);
3. continuous optimiser check of (Q,Q) just above each gamma*;
4. dephasing noise at which (Q,Q) stops being a NE at gamma=pi/2; amplitude-damping gamma* at p=0.6."""
from qgames import *

KINDS = ["depolarizing", "dephasing", "amplitude"]
def grid(nt, nf):
    return np.array([U(t, f) for t in np.linspace(0, np.pi, nt) for f in np.linspace(0, np.pi / 2, nf)])
def nash(S, g, k, p, G):
    return payoffs_A(G, S, g, k, p).max() <= ewl(S, S, g, k, p)[0] + 1e-9
def gstar(k, p, G, tol=1e-5):
    lo, hi = 0.0, np.pi / 2
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if nash(Q, mid, k, p, G) else (mid, hi)
    return (lo + hi) / 2

grids = {n: grid(*n) for n in [(41, 21), (121, 61), (401, 201)]}
print("1. gamma* by grid size")
for k in KINDS:
    for p in (0.1, 0.3):
        gs = [gstar(k, p, G) for G in grids.values()]
        print(f"  {k:13s} p={p}: " + "  ".join(f"{g:.4f}" for g in gs))
        assert max(gs) - min(gs) < 1e-3

print("2. single transition along gamma (400 points)")
for k in KINDS:
    for p in (0.1, 0.3, 0.6):
        v = [nash(Q, g, k, p, GRID) for g in np.linspace(0, np.pi / 2, 400)]
        switches = sum(a != b for a, b in zip(v, v[1:]))
        print(f"  {k:13s} p={p}: switches={switches}"); assert switches == 1

print("3. continuous check just above gamma*")
for k in KINDS:
    for p in (0.1, 0.3):
        g = gstar(k, p, GRID) + 1e-3
        gain = best_deviation(Q, g, k, p)[0] - ewl(Q, Q, g, k, p)[0]
        print(f"  {k:13s} p={p}: best deviation gain = {gain:.1e}"); assert gain < 1e-9

lo, hi = 0.6, 0.7
while hi - lo > 1e-5:
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if nash(Q, np.pi / 2, "dephasing", mid, GRID) else (lo, mid)
print(f"4. dephasing: (Q,Q) stops being a NE at gamma=pi/2 at p = {(lo + hi) / 2:.4f}")
print(f"   amplitude damping gamma* at p=0.6: {gstar('amplitude', 0.6, GRID):.4f}")
