#!/usr/bin/env bash
# Re-runs every experiment of the paper (about 1.5-2 h on a laptop CPU),
# then regenerates figures and LaTeX tables.
set -e
cd "$(dirname "$0")/code"
python3 exp_solvers.py          # EXP-A, EXP-B  -> Table III, Figs. 2-3
python3 exp_system.py CD        # EXP-C, EXP-D  -> Figs. 4-6(a)
python3 exp_rho.py              # Fig. 6(b)
python3 exp_security.py EG      # EXP-E, EXP-G  -> Figs. 7, 9
python3 exp_security.py F       # EXP-F         -> Table IV, Fig. 8
python3 exp_k025.py             # Table IV, kappa = 0.25 rows
python3 plot.py                 # -> figs/*.pdf
python3 tables.py               # -> tables/*.tex
