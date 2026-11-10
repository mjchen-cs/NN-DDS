"""Generate all figures of the revised manuscript from results/*.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "summary")
F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3,
                     "legend.fontsize": 7, "lines.linewidth": 1.2, "lines.markersize": 4,
                     "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
W1 = 3.45
CFG = ["1x1", "4x1", "4x2", "4x4"]
LAB = {"1x1": r"SISO $1\times1$", "4x1": r"MISO $4\times1$", "4x2": r"MIMO $4\times2$", "4x4": r"MIMO $4\times4$"}
MK = {"1x1": "o", "4x1": "s", "4x2": "^", "4x4": "D"}
C4 = {"1x1": "#1b9e77", "4x1": "#d95f02", "4x2": "#7570b3", "4x4": "#e7298a"}


def load(n):
    p = os.path.join(R, n)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_conv():
    d = load("expB.json")
    if d is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 2, 1.9))
    sty = {"NN-DDS (c=5)": ("-", "C0"), "NN-DDS (c=1)": ("--", "C1"), "PDGD": (":", "C3")}
    for k, (ls, c) in sty.items():
        it = np.arange(len(d[k]["gap"]))
        ax[0].semilogy(it, np.maximum(d[k]["gap"], 1e-12), ls, color=c, label=k)
        ax[1].semilogy(it, np.maximum(d[k]["viol"], 1e-12), ls, color=c, label=k)
    ax[0].set_xlabel("iteration $k$"); ax[0].set_ylabel(r"$|f(\mathbf{x}_k)-f^\star|/f^\star$")
    ax[1].set_xlabel("iteration $k$"); ax[1].set_ylabel("constraint violation")
    ax[0].legend(); ax[0].set_title("(a) objective gap"); ax[1].set_title("(b) feasibility")
    fig.savefig(os.path.join(F, "fig_conv.pdf")); plt.close(fig)
    # trajectory of decision variables
    fig, ax = plt.subplots(1, 1, figsize=(W1, 1.8))
    k = "NN-DDS (c=5)"
    it = np.arange(len(d[k]["x2"]))
    ax.plot(it, d[k]["x2"], color="C0", label=r"$b_2/b_{\rm task}$")
    ax.axhline(d["x2_star"], color="C0", ls=":", lw=0.8)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(it, d[k]["tu"], color="C2", label=r"$t_u$ [ms]")
    ax2.axhline(d["tu_star"], color="C2", ls=":", lw=0.8)
    ax.set_xlabel("iteration $k$"); ax.set_ylabel(r"$b_2/b_{\rm task}$", color="C0"); ax2.set_ylabel(r"$t_u$ [ms]", color="C2")
    fig.savefig(os.path.join(F, "fig_traj.pdf")); plt.close(fig)


def fig_track():
    d = load("expC.json")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    Ks = [5, 10, 20, 50, 100, 200]
    for fD, c in zip([10, 100, 500], ["C0", "C1", "C3"]):
        w = [d[f"{fD}_{K}"]["warm"] for K in Ks]
        cc = [d[f"{fD}_{K}"]["cold"] for K in Ks]
        ax.loglog(Ks, np.maximum(w, 1e-7), "-o", color=c, label=f"warm, $f_D$={fD} Hz")
        ax.loglog(Ks, np.maximum(cc, 1e-7), "--x", color=c, label=f"cold, $f_D$={fD} Hz")
    ax.set_xlabel("NN-DDS iterations per channel update $K$")
    ax.set_ylabel("mean relative energy gap")
    ax.legend(ncol=2, fontsize=6)
    fig.savefig(os.path.join(F, "fig_track.pdf")); plt.close(fig)


def fig_sys():
    d = load("expD.json")
    if d is None:
        return
    fig, ax = plt.subplots(2, 2, figsize=(W1 * 2, 3.4))
    keys = [("save", "energy saving vs. local [%]", 100), ("frac", r"offloaded fraction $b_2/b_{\rm task}$", 1),
            ("lat", "completion latency [ms]", 1), ("rate", "uplink rate $r_u$ [Mbit/s]", 1)]
    for a, (k, yl, s) in zip(ax.ravel(), keys):
        for c in CFG:
            g = [r["g"] for r in d[c]]; v = [s * r[k] for r in d[c]]
            a.plot(g, v, "-" + MK[c], color=C4[c], label=LAB[c], markevery=2)
        a.set_xlabel(r"mean channel gain $\bar\gamma$ [dB]"); a.set_ylabel(yl)
    ax[0, 0].legend()
    ax[1, 0].axhline(50, color="k", ls=":", lw=0.8); ax[1, 0].text(1, 51, "$L_m$", fontsize=7)
    for a, t in zip(ax.ravel(), "abcd"):
        a.set_title(f"({t})", loc="left", fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(F, "fig_sys.pdf")); plt.close(fig)


def fig_deadline():
    d = load("expD.json"); dr = load("expRho.json")
    if d is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 2, 1.9))
    for kap, c in zip(["0.25", "0.5", "1.0"], ["C0", "C1", "C3"]):
        rows = d["sweep"][kap]
        L = [1e3 * r["Lm"] for r in rows if r["save"] is not None]
        s = [100 * r["save"] for r in rows if r["save"] is not None]
        ax[0].plot(L, s, "-o", color=c, label=rf"$\kappa={kap}$")
    ax[0].set_xlabel("deadline $L_m$ [ms]"); ax[0].set_ylabel("energy saving [%]"); ax[0].legend()
    ax[0].set_title("(a)", loc="left", fontsize=8)
    if dr:
        for key, ls in zip(sorted(dr.keys()), ["-o", "-s", "-^", "--o", "--s", "--^"]):
            kap, g = key.split("_")
            rows = dr[key]
            ax[1].plot([r["rho"] for r in rows], [r["frac"] for r in rows], ls,
                       label=rf"$\kappa$={kap}, {float(g):.0f} dB", markersize=3)
        ax[1].set_xlabel(r"fallback probability $\rho$"); ax[1].set_ylabel(r"offloaded fraction")
        ax[1].legend(fontsize=6, ncol=2); ax[1].set_title("(b)", loc="left", fontsize=8)
    fig.savefig(os.path.join(F, "fig_deadline.pdf")); plt.close(fig)


def fig_roc():
    d = load("expE.json")
    if d is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(W1 * 2, 1.9))
    for N, c in zip(["1", "2", "4"], ["C0", "C1", "C3"]):
        r = d["jam"][N]
        ax[0].semilogx(r["pfa"], r["pd"], "-", color=c, label=f"$N_r$={N}")
        ax[0].semilogx(r["pfa_mc"], r["pd_mc"], "o", color=c, mfc="none", markersize=3)
    ax[0].set_xlabel(r"$P_{\rm FA}$"); ax[0].set_ylabel(r"$P_{\rm D}$"); ax[0].legend(title="JNR=0 dB", title_fontsize=6)
    ax[0].set_title("(a) jamming test", loc="left", fontsize=8)
    for K, c in zip(["1", "4", "8", "16"], ["C0", "C1", "C2", "C3"]):
        r = d["auth"][K]
        ax[1].semilogx(r["pfa"], r["pd"], "-", color=c, label=f"$K$={K}")
        ax[1].semilogx(r["pfa_mc"], r["pd_mc"], "o", color=c, mfc="none", markersize=3)
        q = d["pd_vs_fd"][K]
        ax[2].plot(q["fd"], q["pd"], "-", color=c, label=f"$K$={K}")
    ax[1].set_xlabel(r"$P_{\rm FA}$"); ax[1].set_ylabel(r"$P_{\rm D}$"); ax[1].legend(title=r"$f_D$=50 Hz", title_fontsize=6)
    ax[1].set_title("(b) PHY authentication", loc="left", fontsize=8)
    ax[2].set_xlabel(r"Doppler $f_D$ [Hz] ($\Delta t$=5 ms)"); ax[2].set_ylabel(r"$P_{\rm D}$ at $P_{\rm FA}=10^{-2}$")
    ax[2].legend(); ax[2].set_title("(c) staleness", loc="left", fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(F, "fig_roc.pdf")); plt.close(fig)


def fig_attack():
    d = load("expF.json")
    if d is None:
        return
    sw = d["sweep"]; pis = sorted(sw.keys(), key=float)
    x = [float(p) for p in pis]
    fig, ax = plt.subplots(1, 3, figsize=(W1 * 2, 1.9))
    names = {"local": "local only", "naive": "security-agnostic", "decoupled": "decoupled shield", "proposed": "proposed"}
    st = {"local": "k:", "naive": "C3-s", "decoupled": "C1-^", "proposed": "C0-o"}
    for s in ["local", "naive", "decoupled", "proposed"]:
        ax[0].plot(x, [sw[p][s]["E"] for p in pis], st[s], label=names[s])
        ax[1].plot(x, [100 * sw[p][s]["dv"] for p in pis], st[s], label=names[s])
        ax[2].plot(x, [100 * sw[p][s]["asr"] for p in pis], st[s], label=names[s])
    ax[0].set_ylabel("energy per task [mJ]"); ax[1].set_ylabel("deadline violation [%]"); ax[2].set_ylabel("attack success rate [%]")
    for a, t in zip(ax, "abc"):
        a.set_xlabel(r"attack probability $\pi$"); a.set_title(f"({t})", loc="left", fontsize=8)
    ax[0].set_yscale("log"); ax[0].legend(fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(F, "fig_attack.pdf")); plt.close(fig)


def fig_an():
    d = load("expG.json")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    ax2 = ax.twinx(); ax2.grid(False)
    for c, col in zip(["4x1", "4x2"], [C4["4x1"], C4["4x2"]]):
        phi = [r["phi"] for r in d[c]]
        ax.semilogy(phi, [max(r["intc"], 1e-3) for r in d[c]], "-" + MK[c], color=col, label=LAB[c] + r", $P_{\rm int}$")
        ax2.plot(phi, [r["E"] for r in d[c]], "--" + MK[c], color=col, mfc="none", label=LAB[c] + ", energy")
    ax.set_xlabel(r"AN power fraction $\phi$"); ax.set_ylabel(r"interception prob. $P_{\rm int}$"); ax2.set_ylabel("energy [mJ]")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="center right")
    fig.savefig(os.path.join(F, "fig_an.pdf")); plt.close(fig)


if __name__ == "__main__":
    for fn in [fig_conv, fig_track, fig_sys, fig_deadline, fig_roc, fig_attack, fig_an]:
        fn()
    print("done")
