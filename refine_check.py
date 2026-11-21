"""Continuous check of the grid-based NE test: for (Q,Q) and (D,D) on a (channel, p, gamma) sweep,
a bounded optimiser started from the 5 best grid points never beats the 41x21 grid and never flips a verdict.
Takes about two minutes."""
from qgames import *

worst, flips, n = 0.0, 0, 0
for k in ["depolarizing", "dephasing", "amplitude"]:
    for p in np.linspace(0, 0.6, 13):
        for g in np.linspace(0, np.pi / 2, 25):
            for S in (Q, D):
                grid = payoffs_A(GRID, S, g, k, p).max()
                cont = best_deviation(S, g, k, p)[0]
                base = ewl(S, S, g, k, p)[0]
                worst = max(worst, cont - grid); n += 1
                flips += (grid <= base + 1e-9) != (cont <= base + 1e-9)
print(f"{n} profiles: largest continuous gain over grid = {worst:.1e}, verdict flips = {flips}")
assert worst < 1e-9 and flips == 0
