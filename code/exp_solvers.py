"""EXP-A: solver comparison on (P2);  EXP-B: convergence trajectories."""
import json, time, warnings
import numpy as np
from scipy.stats import wilcoxon
from sklearn.neural_network import MLPRegressor
from model import *
from solvers import *
from rawlog import log, flush, summary
warnings.filterwarnings("ignore")

CONFIGS = [(1, 1), (4, 1), (4, 2), (4, 4)]   # (Nt device, Nr AP)
p = Params(rho=0.02, phi_an=0.1)


def make(nt, nr, gdb, rng):
    H = rayleigh(nr, nt, rng, 10 ** (gdb / 10))
    lam = eig_gains(H)
    L = Link(p, lam, lam, n_an_dim=max(nt - nr, 0))
    return Problem(L)


def features(pb, n=4):
    lam = np.zeros(n); lam[:len(pb.L.lam)] = np.log10(pb.L.lam)
    return np.r_[lam, np.log10(pb.L.r_max), np.log10(pb.L.Rd)]


def l2o_project(pb, z):
    p, L = pb.p, pb.L
    lo, hi = pb.feasible()
    x2 = float(np.clip(z[0], lo, hi))
    B, Lm = p.btask, p.Lm
    td = p.beta_d * B * x2 / L.Rd
    tmax = Lm - p.tv - (p.tau2 + p.kappa * p.tau1) * B * x2 - td
    tlo = p.beta_u * B * x2 / L.r_max
    tu = float(np.clip(z[1] * Lm, tlo, tmax))
    return np.array([1 - x2, x2, tu / Lm, td / Lm])


def run_A(nchan=200, seed=7):
    rng = np.random.default_rng(seed)
    out = {}
    for nt, nr in CONFIGS:
        # ---- train learning-to-optimize MLP on separate channels
        Xtr, Ytr = [], []
        while len(Xtr) < 3000:
            pb = make(nt, nr, rng.uniform(5, 35), rng)
            if pb.feasible() is None:
                continue
            x = bisection(pb)["x"]
            Xtr.append(features(pb)); Ytr.append([x[1], x[2]])
        mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=0)
        mlp.fit(np.array(Xtr), np.array(Ytr))
        res = {k: dict(gap=[], viol=[], time=[], it=[]) for k in
               ["IPM", "Bisection", "SLSQP", "AO", "PDGD", "PSO", "L2O", "NN-DDS"]}
        cnt = 0
        while cnt < nchan:
            gdb = rng.uniform(5, 35)
            pb = make(nt, nr, gdb, rng)
            if pb.feasible() is None:
                continue
            cnt += 1
            r = ipm(pb); f_ref = pb.f(r["x"])[0]
            res["IPM"]["time"].append(r["time"]); res["IPM"]["it"].append(r["it"] or 0)
            res["IPM"]["gap"].append(0.0); res["IPM"]["viol"].append(pb.viol(r["x"]))
            for name, fn in [("Bisection", bisection), ("SLSQP", slsqp), ("AO", ao), ("PDGD", pdgd),
                             ("PSO", pso), ("NN-DDS", nndds)]:
                t0 = time.perf_counter(); r = fn(pb); dt = time.perf_counter() - t0
                x = r["x"]
                res[name]["gap"].append((pb.f(x)[0] - f_ref) / f_ref)
                res[name]["viol"].append(pb.viol(x)); res[name]["time"].append(dt)
                res[name]["it"].append(r.get("it", 0))
                log("expA_solvers", config=f"{nt}x{nr}", instance=cnt, gain_dB=gdb, solver=name,
                    energy_norm=pb.f(x)[0], energy_ipm_norm=f_ref, rel_gap=res[name]["gap"][-1],
                    violation=res[name]["viol"][-1], time_s=dt, iterations=res[name]["it"][-1],
                    b2_frac=x[1], tu_s=x[2] * pb.p.Lm, td_s=x[3] * pb.p.Lm)
            t0 = time.perf_counter()
            x = l2o_project(pb, mlp.predict(features(pb)[None])[0]); dt = time.perf_counter() - t0
            res["L2O"]["gap"].append((pb.f(x)[0] - f_ref) / f_ref)
            res["L2O"]["viol"].append(pb.viol(x)); res["L2O"]["time"].append(dt); res["L2O"]["it"].append(1)
            log("expA_solvers", config=f"{nt}x{nr}", instance=cnt, gain_dB=gdb, solver="L2O",
                energy_norm=pb.f(x)[0], energy_ipm_norm=f_ref, rel_gap=res["L2O"]["gap"][-1],
                violation=res["L2O"]["viol"][-1], time_s=dt, iterations=1,
                b2_frac=x[1], tu_s=x[2] * pb.p.Lm, td_s=x[3] * pb.p.Lm)
            log("expA_solvers", config=f"{nt}x{nr}", instance=cnt, gain_dB=gdb, solver="IPM",
                energy_norm=f_ref, energy_ipm_norm=f_ref, rel_gap=0.0, violation=res["IPM"]["viol"][-1],
                time_s=res["IPM"]["time"][-1], iterations=res["IPM"]["it"][-1])
        summ = {}
        nn = np.array(res["NN-DDS"]["gap"])
        for k, v in res.items():
            g = np.array(v["gap"])
            pval = None
            if k not in ("NN-DDS", "IPM") and np.any(np.abs(g - nn) > 0):
                pval = float(wilcoxon(np.abs(g), np.abs(nn), alternative="greater").pvalue)
            summ[k] = dict(gap_mean=float(np.mean(g)), gap_p95=float(np.percentile(g, 95)),
                           gap_max=float(np.max(g)), viol_max=float(np.max(v["viol"])),
                           time_ms=float(1e3 * np.mean(v["time"])), it=float(np.mean(v["it"])), p=pval)
        out[f"{nt}x{nr}"] = summ
        print(nt, nr, json.dumps(summ, indent=0)[:2000])
    json.dump(out, open(summary("expA.json"), "w"), indent=1)
    flush("expA_solvers")
    return out


def run_B(seed=3):
    rng = np.random.default_rng(seed)
    pb = make(4, 2, 18.0, rng)
    fs = pb.f(ipm(pb)["x"])[0]
    curves = {}
    for name, fn in [("NN-DDS (c=5)", lambda q: nndds(q, iters=600, tol=0, record=True)),
                     ("NN-DDS (c=1)", lambda q: nndds(q, c=1.0, eta=4.0, iters=600, tol=0, record=True)),
                     ("PDGD", lambda q: pdgd(q, iters=600, record=True))]:
        r = fn(pb)
        curves[name] = dict(gap=[abs(h[1] - fs) / fs for h in r["hist"]], viol=[h[2] for h in r["hist"]],
                            x2=[h[3][1] for h in r["hist"]], tu=[h[3][2] * pb.p.Lm * 1e3 for h in r["hist"]])
    curves["x2_star"] = float(ipm(pb)["x"][1]); curves["tu_star"] = float(ipm(pb)["x"][2] * pb.p.Lm * 1e3)
    json.dump(curves, open(summary("expB.json"), "w"))
    for name in ["NN-DDS (c=5)", "NN-DDS (c=1)", "PDGD"]:
        for k, (g, v, x2, tu) in enumerate(zip(*[curves[name][q] for q in ("gap", "viol", "x2", "tu")])):
            log("expB_convergence", method=name, iteration=k, rel_gap=g, violation=v, b2_frac=x2, tu_ms=tu)
    flush("expB_convergence")


if __name__ == "__main__":
    run_B()
    run_A()
