"""EXP-C: tracking of time-varying channels;  EXP-D: system-level behaviour."""
import json, time, warnings
import numpy as np
from dataclasses import replace
from scipy.special import j0
from model import *
from solvers import *
from exp_solvers import l2o_project
from rawlog import log, flush, summary
warnings.filterwarnings("ignore")

CONFIGS = [(1, 1), (4, 1), (4, 2), (4, 4)]
P0 = Params(rho=0.02, phi_an=0.1)


def prob_from_H(p, H):
    nr, nt = H.shape
    lam = eig_gains(H)
    return Problem(Link(p, lam, lam, n_an_dim=max(nt - nr, 0)))


def solve_safe(pb, **kw):
    """NN-DDS followed by the (cheap) feasibility restoration used at run time."""
    r = nndds(pb, **kw)
    return l2o_project(pb, [r["x"][1], r["x"][2]]), r


def run_C(seed=11, nupd=400, delta=1e-3):
    rng = np.random.default_rng(seed)
    out = {}
    for fD in [10, 100, 500]:
        rho = j0(2 * np.pi * fD * delta)
        seqrng = np.random.default_rng(seed + fD)
        Hs = [rayleigh(2, 4, seqrng, 10 ** 2.0)]
        for _ in range(nupd):
            Hs.append(rho * Hs[-1] + np.sqrt(1 - rho ** 2) * rayleigh(2, 4, seqrng, 10 ** 2.0))
        for K in [5, 10, 20, 50, 100, 200]:
            x, lam = None, None
            gw, gc, vw = [], [], []
            for k in range(nupd):
                pb = prob_from_H(P0, Hs[k + 1])
                if pb.feasible() is None:
                    continue
                fs = pb.f(bisection(pb)["x"])[0]
                r = nndds(pb, x0=x, lam0=lam, iters=K, tol=0, xbar=x)
                x, lam = r["x"], r["lam"]
                vw.append(pb.viol(x))
                xs = l2o_project(pb, [x[1], x[2]])
                gw.append((pb.f(xs)[0] - fs) / fs)
                rc = nndds(pb, iters=K, tol=0)
                xc = l2o_project(pb, [rc["x"][1], rc["x"][2]])
                gc.append((pb.f(xc)[0] - fs) / fs)
                log("expC_tracking", fD_Hz=fD, K=K, update=k, energy_opt_norm=fs,
                    gap_warm=gw[-1], gap_cold=gc[-1], viol_warm_raw=vw[-1])
            out[f"{fD}_{K}"] = dict(fD=fD, K=K, warm=float(np.mean(gw)), cold=float(np.mean(gc)),
                                    warm95=float(np.percentile(gw, 95)), viol=float(np.mean(vw)))
            print(out[f"{fD}_{K}"])
    # time per Euler iteration (python, this machine)
    pb = prob_from_H(P0, rayleigh(2, 4, rng, 100.0))
    t0 = time.perf_counter(); nndds(pb, iters=2000, tol=0); t_it = (time.perf_counter() - t0) / 2000
    out["t_iter_us"] = 1e6 * t_it
    json.dump(out, open(summary("expC.json"), "w"), indent=1)
    flush("expC_tracking")


def run_D(seed=5, nch=200):
    rng = np.random.default_rng(seed)
    gdbs = np.arange(0, 41, 2.5)
    res = {}
    for nt, nr in CONFIGS:
        rows = []
        for g in gdbs:
            acc = dict(save=[], frac=[], lat=[], rate=[], feas=0)
            for _ in range(nch):
                pb = prob_from_H(P0, rayleigh(nr, nt, rng, 10 ** (g / 10)))
                if pb.feasible() is None:
                    continue
                acc["feas"] += 1
                x, _ = solve_safe(pb)
                p = pb.p
                acc["save"].append(1 - pb.energy(x) / pb.Eref)
                acc["frac"].append(x[1])
                tl = p.tau1 * p.btask * x[0]
                to = (x[2] + x[3]) * p.Lm + p.tau2 * p.btask * x[1] + p.tv if x[1] > 1e-6 else 0
                acc["lat"].append(1e3 * max(tl, to))
                acc["rate"].append(p.beta_u * p.btask * x[1] / (x[2] * p.Lm) / 1e6 if x[1] > 1e-6 else 0)
                log("expD_gain", config=f"{nt}x{nr}", gain_dB=float(g), energy_J=pb.energy(x),
                    saving=acc["save"][-1], b2_frac=x[1], latency_ms=acc["lat"][-1], rate_Mbps=acc["rate"][-1],
                    tu_s=x[2] * p.Lm, td_s=x[3] * p.Lm)
            rows.append(dict(g=float(g), **{k: float(np.mean(v)) for k, v in acc.items() if k != "feas"},
                             rmax=0.0, feas=acc["feas"] / nch))
        res[f"{nt}x{nr}"] = rows
        print(nt, nr, [(r["g"], round(r["save"], 3), round(r["frac"], 3)) for r in rows])
    # sweep of deadline and fallback cost at 4x2, 20 dB
    sweep = {}
    for kappa in [0.25, 0.5, 1.0]:
        rows = []
        for Lm in np.arange(0.02, 0.105, 0.005):
            p = replace(P0, kappa=kappa, Lm=float(Lm))
            s, fr, n = [], [], 0
            r2 = np.random.default_rng(99)
            for _ in range(nch):
                pb = prob_from_H(p, rayleigh(2, 4, r2, 100.0))
                if pb.feasible() is None:
                    continue
                x, _ = solve_safe(pb)
                s.append(1 - pb.energy(x) / pb.Eref); fr.append(x[1]); n += 1
                log("expD_deadline", kappa=kappa, Lm_s=float(Lm), energy_J=pb.energy(x), saving=s[-1], b2_frac=x[1])
            rows.append(dict(Lm=float(Lm), save=float(np.mean(s)) if s else None,
                             frac=float(np.mean(fr)) if fr else None, feas=n / nch))
        sweep[str(kappa)] = rows
    res["sweep"] = sweep
    # energy vs fallback probability rho (4x2, 20 dB)
    rr = []
    for rho in [0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        p = replace(P0, rho=rho)
        r2 = np.random.default_rng(42)
        s, fr = [], []
        for _ in range(nch):
            pb = prob_from_H(p, rayleigh(2, 4, r2, 100.0))
            x, _ = solve_safe(pb)
            s.append(pb.energy(x)); fr.append(x[1])
        rr.append(dict(rho=rho, E=float(np.mean(s)), frac=float(np.mean(fr))))
    res["rho"] = rr
    json.dump(res, open(summary("expD.json"), "w"), indent=1)
    flush("expD_gain"); flush("expD_deadline")


if __name__ == "__main__":
    import sys
    w = sys.argv[1] if len(sys.argv) > 1 else "CD"
    if "C" in w: run_C()
    if "D" in w: run_D()
