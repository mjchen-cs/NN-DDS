"""EXP-E: detector ROC (closed form vs Monte Carlo)
   EXP-F: end-to-end attack scenarios (energy, deadline violation, attack success, interception)
   EXP-G: artificial-noise privacy/energy trade-off."""
import json, warnings
import numpy as np
from dataclasses import replace
from model import *
from solvers import nndds
from exp_solvers import l2o_project
from security import *
from rawlog import log, flush, summary
warnings.filterwarnings("ignore")

P0 = Params(rho=0.02, phi_an=0.1)
LP = 16            # pilot length
PFA = 1e-2         # target false-alarm probability of each test
JNR = 10.0         # 10 dB jammer-to-noise ratio at the AP
FD, DT = 10.0, 5e-3  # Doppler and reference-refresh interval for PHY authentication
D_CMD, A_OFF = 4, 16.0  # command dimension, attacker offset ||a||^2/sigma_l^2 (4 sigma)
NE = 2             # eavesdropper antennas


# -------------------------------------------------------------- EXP-E
def run_E(seed=1, n=20000):
    rng = np.random.default_rng(seed)
    out = dict(jam={}, auth={}, pd_vs_fd={})
    pfas = np.logspace(-4, 0, 25)
    for N in [1, 2, 4]:
        h0 = np.array([jam_stat_mc(N, LP, 0.0, rng) for _ in range(n)])
        h1 = np.array([jam_stat_mc(N, LP, 10 ** 0.0, rng) for _ in range(n)])   # JNR 0 dB (weak)
        for i in range(n):
            log("expE_jam_stats", Nr=N, trial=i, T_H0=h0[i], T_H1=h1[i])
        th = [jam_threshold(N, LP, a) for a in pfas]
        out["jam"][N] = dict(pfa=pfas.tolist(), pd=[float(jam_pd(N, LP, a, 1.0)) for a in pfas],
                             pfa_mc=[float(np.mean(h0 > t)) for t in th], pd_mc=[float(np.mean(h1 > t)) for t in th])
    se2 = 1 / (LP * 100.0 / 2)
    for K in [1, 4, 8, 16]:
        s0, s1 = auth_params(K, 50.0, DT, se2)
        h0 = np.array([auth_stat_mc(K, True, 50.0, DT, se2, rng) for _ in range(n)])
        h1 = np.array([auth_stat_mc(K, False, 50.0, DT, se2, rng) for _ in range(n)])
        for i in range(n):
            log("expE_auth_stats", K=K, trial=i, T_legit=h0[i], T_spoof=h1[i])
        th = [auth_threshold(K, a) for a in pfas]
        out["auth"][K] = dict(pfa=pfas.tolist(), pd=[float(auth_pd(K, a, s0, s1)) for a in pfas],
                              pfa_mc=[float(np.mean(h0 > t)) for t in th], pd_mc=[float(np.mean(h1 > t)) for t in th])
    fds = np.linspace(1, 120, 60)
    for K in [1, 4, 8, 16]:
        out["pd_vs_fd"][K] = dict(fd=fds.tolist(),
                                  pd=[float(auth_pd(K, PFA, *auth_params(K, f, DT, se2))) for f in fds])
    json.dump(out, open(summary("expE.json"), "w"))
    flush("expE_jam_stats"); flush("expE_auth_stats")


# -------------------------------------------------------------- EXP-F
def plan(p, H, R=None, rng=None):
    nr, nt = H.shape
    lam = eig_gains(H, R)
    pb = Problem(Link(p, lam, eig_gains(H.T), n_an_dim=max(nt - nr, 0)))
    if pb.feasible() is None:
        return pb, np.array([1.0, 0.0, 0.0, 0.0])
    r = nndds(pb)
    return pb, l2o_project(pb, [r["x"][1], r["x"][2]])


def covariance(H, R, P_sig):
    """Signal covariance designed on the (whitened) channel H^H R^-1 H."""
    nt = H.shape[1]
    M = H.conj().T @ (H if R is None else np.linalg.solve(R, H))
    w, V = np.linalg.eigh(M); idx = np.argsort(w)[::-1]; w, V = w[idx], V[:, idx]
    r = int(np.sum(w > 1e-9 * w[0])); lam = w[:r]
    for G in range(r, 0, -1):
        mu = (P_sig + np.sum(1 / lam[:G])) / G
        if mu > 1 / lam[G - 1]:
            break
    pw = np.maximum(mu - 1 / lam, 0)
    return (V[:, :r] * pw) @ V[:, :r].conj().T


def null_an(H, P_an):
    nt = H.shape[1]
    _, s, Vh = np.linalg.svd(H); r = int(np.sum(s > 1e-9))
    if nt <= r or P_an <= 0:
        return np.zeros((nt, nt), complex)
    Vn = Vh.conj().T[:, r:]
    return Vn @ Vn.conj().T * P_an / (nt - r)


def run_round(scheme, p, nt, nr, gdb, ev, rng, rho_hat):
    """Returns dict(E, dv, asr, intc, fb, frac)."""
    g = 10 ** (gdb / 10)
    H, hJ, Ge = ev["H"], ev["hJ"], ev["Ge"]
    Rj = np.eye(nr) + JNR * np.outer(hJ, hJ.conj())
    jammed_plan, jammed_run = ev["persist"], ev["persist"] or ev["react"]
    shield = scheme in ("decoupled", "proposed")
    if scheme == "local":
        return dict(E=p.eps_loc * p.btask, dv=0, asr=0, intc=0, fb=0, frac=0.0, det=None)
    if scheme == "proposed":
        pp = replace(p, rho=rho_hat)
    else:   # security-agnostic plan: original problem without window / rho / AN
        pp = replace(p, rho=0.0, phi_an=0.0, kappa=0.0, tv=0.0)
    # --- AP-side pilot residual test for (persistent) jamming before planning
    det = None
    if shield:
        stat = jam_stat_mc(nr, LP, JNR * np.linalg.norm(hJ) ** 2 / nr if jammed_plan else 0.0, rng)
        det = stat > jam_threshold(nr, LP, PFA)
    Rplan = Rj if (scheme == "proposed" and det and jammed_plan) else None
    pb, x = plan(pp, H, Rplan)
    B, Lm = p.btask, p.Lm
    b1, b2 = x[0] * B, x[1] * B
    E = p.eps_loc * b1
    if b2 <= 1e-9:
        return dict(E=E, dv=0, asr=0, intc=0, fb=0, frac=0.0, det=det)
    tu, td = x[2] * Lm, x[3] * Lm
    ru = p.beta_u * b2 / tu
    Eu, _, _ = pb.L.Eu(p.beta_u * b2, tu)
    E += Eu
    # --- uplink success with interference-rejection combining at the AP
    Psig, _, _ = pb.L.P(ru)
    Q = covariance(H, Rplan, Psig)
    se_act = bob_rate_irc(H, Q, JNR if jammed_run else 0.0, hJ / max(np.linalg.norm(hJ), 1e-12) * np.linalg.norm(hJ))
    up_ok = se_act >= ru / p.Wu * (1 - 1e-6)
    # --- interception by a passive multi-antenna eavesdropper
    Qan = null_an(H, pb.L.P_an)
    intc = int(eve_rate(Ge, Q, Qan) >= ru / p.Wu)
    t_resp = tu + p.tau2 * b2 + td + (p.tv if shield else 0.0)
    win = Lm - p.kappa * p.tau1 * b2 if shield else np.inf   # derived safe waiting window
    asr = 0; accepted = False; fb = 0
    # forged response (arrives first if present)
    if ev["spoof"]:
        if not shield:
            accepted, asr = True, 1
        else:
            se2 = 1 / (LP * g * p.Pa / nr)
            K = nt * nr
            s0, _ = auth_params(K, FD, DT, se2)
            pass_auth = auth_stat_mc(K, False, FD, DT, se2, rng) <= auth_threshold(K, PFA)
            off = rng.chisquare(D_CMD) if False else None
            pass_cons = rng.random() > cons_pd(D_CMD, PFA, A_OFF)
            if pass_auth and pass_cons:
                accepted, asr = True, 1
    if not accepted:
        legit_ok = up_ok and t_resp <= win + 1e-9
        if legit_ok and shield:
            se2 = 1 / (LP * g * p.Pa / nr)
            K = nt * nr
            fa = (auth_stat_mc(K, True, FD, DT, se2, rng) > auth_threshold(K, PFA)) or (rng.random() < PFA)
            legit_ok = not fa
        if legit_ok:
            E += p.pd * td + p.alpha_d * p.beta_d * b2
            dv = int(t_resp > Lm)
        else:
            if not shield:
                dv = 1          # no valid response before the deadline
            else:
                fb = 1
                t_f = min(t_resp, win) if up_ok else win
                E += p.kappa * p.eps_loc * b2
                dv = int(max(t_f, p.tau1 * b1) + p.kappa * p.tau1 * b2 > Lm + 1e-12)
    else:
        E += p.pd * td + p.alpha_d * p.beta_d * b2
        dv = 0
    return dict(E=E, dv=dv, asr=asr, intc=intc, fb=fb, frac=x[1], det=det)


def scenario(nt, nr, gdb, pis, nround, seed, schemes=("local", "naive", "decoupled", "proposed"), p=P0, tag=None):
    rng = np.random.default_rng(seed)
    evs = []
    for _ in range(nround):
        evs.append(dict(H=rayleigh(nr, nt, rng, 10 ** (gdb / 10)), hJ=rayleigh(nr, 1, rng).ravel(),
                        Ge=rayleigh(NE, nt, rng, 10 ** (gdb / 10)),
                        persist=rng.random() < pis[0], react=rng.random() < pis[1], spoof=rng.random() < pis[2]))
    out = {}
    for s in schemes:
        r2 = np.random.default_rng(seed + 1)
        nfb, acc = 0, []
        for k, ev in enumerate(evs):
            rho_hat = (1 + nfb) / (21 + k)          # Beta(1,20) prior, online update
            r = run_round(s, p, nt, nr, gdb, ev, r2, rho_hat)
            nfb += r["fb"]; acc.append(r)
            if tag:
                log(tag, config=f"{nt}x{nr}", gain_dB=gdb, pi_persist=pis[0], pi_react=pis[1], pi_spoof=pis[2],
                    kappa=p.kappa, phi_an=p.phi_an, scheme=s, round=k, ev_persist=int(ev["persist"]),
                    ev_react=int(ev["react"]), ev_spoof=int(ev["spoof"]), rho_hat=rho_hat, energy_J=r["E"],
                    deadline_violation=r["dv"], attack_success=r["asr"], intercepted=r["intc"],
                    fallback=r["fb"], b2_frac=r["frac"], jam_detected=("" if r["det"] is None else int(r["det"])))
        jam_det = [a["det"] for a, e in zip(acc, evs) if a["det"] is not None and e["persist"]]
        jam_fa = [a["det"] for a, e in zip(acc, evs) if a["det"] is not None and not e["persist"]]
        out[s] = dict(E=1e3 * float(np.mean([a["E"] for a in acc])),
                      dv=float(np.mean([a["dv"] for a in acc])), asr=float(np.mean([a["asr"] for a in acc])),
                      intc=float(np.mean([a["intc"] for a in acc])), fb=float(np.mean([a["fb"] for a in acc])),
                      frac=float(np.mean([a["frac"] for a in acc])),
                      pd_jam=float(np.mean(jam_det)) if jam_det else None,
                      pfa_jam=float(np.mean(jam_fa)) if jam_fa else None)
    return out


def run_F(nround=1500):
    res = {}
    for nt, nr in [(1, 1), (4, 1), (4, 2), (4, 4)]:
        res[f"{nt}x{nr}"] = scenario(nt, nr, 20.0, (0.1, 0.1, 0.1), nround, seed=100 + nt * 10 + nr, tag="expF_rounds")
        print(nt, nr, json.dumps(res[f"{nt}x{nr}"]))
    sweep = {}
    for pi in [0.0, 0.05, 0.1, 0.2, 0.3]:
        sweep[str(pi)] = scenario(4, 2, 20.0, (pi, pi, pi), 1000, seed=500, tag="expF_sweep_rounds")
        print(pi, json.dumps(sweep[str(pi)]))
    res["sweep"] = sweep
    json.dump(res, open(summary("expF.json"), "w"), indent=1)
    flush("expF_rounds"); flush("expF_sweep_rounds")


def run_G(nround=1000):
    res = {}
    for nt, nr in [(4, 1), (4, 2)]:
        rows = []
        for phi in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
            p = replace(P0, phi_an=phi)
            o = scenario(nt, nr, 20.0, (0, 0, 0), nround, seed=900, schemes=("proposed",), p=p, tag="expG_rounds")["proposed"]
            rows.append(dict(phi=phi, **o))
        res[f"{nt}x{nr}"] = rows
        print(nt, nr, rows)
    json.dump(res, open(summary("expG.json"), "w"), indent=1)
    flush("expG_rounds")


if __name__ == "__main__":
    import os, sys
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    which = sys.argv[1] if len(sys.argv) > 1 else "EFG"
    if "E" in which: run_E()
    if "G" in which: run_G()
    if "F" in which: run_F()
