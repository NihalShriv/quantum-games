# Quantum games under noise: CHSH vs the quantum Prisoner's Dilemma

End-semester project, Quantum Information Theory, BITS Goa.

## Reproduce
```
pip install -r requirements.txt
python qgames.py            # self-checks, prints thresholds, writes all figures in figures/ (about one minute)
python qiskit_circuits.py   # Qiskit circuits checked against the numpy model
python mixed_ne.py          # 5/2 mixed Nash equilibrium (full SU(2)), a few minutes
python focal_ne.py          # 9/4 focal (weak) equilibrium
python refine_check.py      # optimiser check that the NE grid misses nothing
```
Report: `report/main.tex`. Build it with `latexmk -pdf main.tex` inside `report/`.

## Files
| file | what |
|---|---|
| `qgames.py` | exact density-matrix model: CHSH (fixed and optimised settings), EWL, Nash check, the course noise channels (depolarizing, dephasing, amplitude damping); prints the CHSH and EWL thresholds |
| `qiskit_circuits.py` | same games as Qiskit circuits (`J = RYYGate(gamma)`) |
| `mixed_ne.py`, `focal_ne.py` | mixed-equilibrium checks in full SU(2) |
| `refine_check.py` | continuous optimiser check of the grid-based NE test (about two minutes) |
| `figures/` | generated figures used in the report |
| `report/` | `main.tex` and the built `main.pdf` |
| `requirements.txt` | pinned Python dependencies |

## Conventions
- Payoff order CC, CD, DC, DD with Alice first in numpy (`np.kron(UA, UB)`).
- Qiskit bitstrings are little-endian (`'Bob Alice'`), so CD = `'10'` and DC = `'01'`.
- Do not transpile noisy circuits at `optimization_level >= 1`, because the noise gets optimised away.
