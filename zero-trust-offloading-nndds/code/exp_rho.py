"""Effect of the fallback probability rho on the offloading decision (4x2)."""
import json, warnings
import numpy as np
from dataclasses import replace
from exp_system import P0, prob_from_H, solve_safe
from model import rayleigh
from rawlog import log, flush, summary
warnings.filterwarnings("ignore")
out = {}
for kappa in [0.5, 1.0]:
    for g in [5.0, 10.0, 20.0]:
        rows = []
        for rho in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            p = replace(P0, rho=rho, kappa=kappa)
            r2 = np.random.default_rng(42)
            E, fr = [], []
            for _ in range(150):
                pb = prob_from_H(p, rayleigh(2, 4, r2, 10 ** (g / 10)))
                if pb.feasible() is None:
                    continue
                x, _ = solve_safe(pb)
                E.append(pb.energy(x)); fr.append(x[1])
                log("expRho", kappa=kappa, gain_dB=g, rho=rho, energy_J=E[-1], b2_frac=x[1])
            rows.append(dict(rho=rho, E=float(np.mean(E)), frac=float(np.mean(fr))))
        out[f"{kappa}_{g}"] = rows
        print(kappa, g, [(r["rho"], round(r["frac"], 3)) for r in rows])
json.dump(out, open(summary("expRho.json"), "w"), indent=1)
flush("expRho")
