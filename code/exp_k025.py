import json, os
from dataclasses import replace
import exp_security as e
from rawlog import summary
os.chdir(os.path.dirname(os.path.abspath(__file__)))
res = {}
for nt, nr in [(1, 1), (4, 1), (4, 2), (4, 4)]:
    p = replace(e.P0, kappa=0.25)
    res[f"{nt}x{nr}"] = e.scenario(nt, nr, 20.0, (0.1, 0.1, 0.1), 1500, seed=100 + nt * 10 + nr,
                                   schemes=("proposed",), p=p, tag="expF_k025_rounds")["proposed"]
    print(nt, nr, res[f"{nt}x{nr}"])
json.dump(res, open(summary("expF_k025.json"), "w"), indent=1)
e.flush("expF_k025_rounds")
