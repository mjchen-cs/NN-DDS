"""Write LaTeX tables from results/*.json (avoids manual transcription)."""
import json, os
import numpy as np
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "summary")
T = os.path.join(B, "tables")
os.makedirs(T, exist_ok=True)


def sci(v, d=1):
    if v is None:
        return "--"
    if v == 0:
        return "0"
    s = f"{v:.{d}e}"
    m, e = s.split("e")
    return f"${m}\\!\\times\\!10^{{{int(e)}}}$"


def tab_solvers():
    d = json.load(open(os.path.join(R, "expA.json")))
    algs = ["IPM", "Bisection", "SLSQP", "AO", "PDGD", "PSO", "L2O", "NN-DDS"]
    names = {"IPM": "IPM (Clarabel)", "Bisection": "Struct. bisection", "SLSQP": "SQP (SLSQP)",
             "AO": "AO", "PDGD": "PDGD", "PSO": "PSO", "L2O": "L2O (MLP)", "NN-DDS": "NN-DDS"}
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Solver comparison on (P2) over 200 channel realizations per configuration ($\bar\gamma\sim\mathcal{U}[5,35]$\,dB). Gap: relative energy gap to the IPM solution (95th percentile); Viol.: maximum constraint violation; Time: mean wall-clock time per solve (Python, single thread); It.: mean iterations (PSO: generations of 40 particles; IPM: Newton steps). $p$: one-sided Wilcoxon signed-rank test that the absolute gap exceeds that of NN-DDS.}",
         r"\label{tab:solvers}", r"\setlength{\tabcolsep}{3.2pt}", r"\footnotesize",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{l" + "rrrr" * 2 + "}", r"\toprule",
         r" & \multicolumn{4}{c}{SISO $1\times1$} & \multicolumn{4}{c}{MIMO $4\times2$}\\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Solver & Gap (p95) & Viol. & Time [ms] & It. & Gap (p95) & Viol. & Time [ms] & It.\\", r"\midrule"]
    for a in algs:
        row = [names[a]]
        for c in ["1x1", "4x2"]:
            s = d[c][a]
            g = max(s["gap_p95"], 0.0)
            row += ["0 (ref.)" if a == "IPM" else sci(g), sci(s["viol_max"]), f"{s['time_ms']:.1f}", f"{s['it']:.0f}"]
        L.append(" & ".join(row) + r"\\")
    L += [r"\midrule", r"\multicolumn{9}{p{1.05\textwidth}}{Wilcoxon $p$-values (all four configurations, $1\times1$/$4\times1$/$4\times2$/$4\times4$):\ " +
          "; ".join(f"{names[a]}: " + "/".join(("--" if d[c][a]['p'] is None else (f"{d[c][a]['p']:.0e}" if d[c][a]['p'] < 0.01 else f"{d[c][a]['p']:.2f}")) for c in ["1x1", "4x1", "4x2", "4x4"]) for a in ["AO", "PDGD", "PSO", "L2O"]) + r"}\\",
          r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(T, "tab_solvers.tex"), "w").write("\n".join(L))


def tab_attack():
    d = json.load(open(os.path.join(R, "expF.json")))
    k25 = json.load(open(os.path.join(R, "expF_k025.json"))) if os.path.exists(os.path.join(R, "expF_k025.json")) else {}
    names = {"local": "Local only", "naive": "Security-agnostic", "decoupled": "Decoupled shield",
             "proposed": r"Proposed ($\kappa=0.5$)", "proposed25": r"Proposed ($\kappa=0.25$)"}
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{End-to-end trust evaluation at $\bar\gamma=20$\,dB with $\pi_P=\pi_R=\pi_S=0.1$ (1500 rounds per configuration). E: mean device energy per task; DVP: deadline violation probability; ASR: attack success rate; $P_{\rm int}$: interception probability; FB: fallback rate; $\hat P_{\rm D}^J/\hat P_{\rm FA}^J$: empirical jamming-detection/false-alarm rates.}",
         r"\label{tab:attack}", r"\setlength{\tabcolsep}{2.6pt}", r"\footnotesize",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{ll rrrrrr r}", r"\toprule",
         r"Config. & Scheme & E [mJ] & DVP [\%] & ASR [\%] & $P_{\rm int}$ [\%] & FB [\%] & $b_2/b_{\rm task}$ & $\hat P_{\rm D}^J/\hat P_{\rm FA}^J$\\", r"\midrule"]
    labs = {"1x1": r"SISO $1\times1$", "4x1": r"MISO $4\times1$", "4x2": r"MIMO $4\times2$", "4x4": r"MIMO $4\times4$"}
    for c in ["1x1", "4x1", "4x2", "4x4"]:
        rows = [(s, d[c][s]) for s in ["local", "naive", "decoupled", "proposed"]]
        if c in k25:
            rows.append(("proposed25", k25[c]))
        for i, (s, v) in enumerate(rows):
            det = "--" if v["pd_jam"] is None else f"{v['pd_jam']:.2f}/{v['pfa_jam']:.3f}"
            L.append((rf"\multirow{{{len(rows)}}}{{*}}{{{labs[c]}}}" if i == 0 else "") +
                     f" & {names[s]} & {v['E']:.1f} & {100*v['dv']:.1f} & {100*v['asr']:.2f} & {100*v['intc']:.1f} & {100*v['fb']:.1f} & {v['frac']:.2f} & {det}\\\\")
        L.append(r"\midrule" if c != "4x4" else r"\bottomrule")
    L += [r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(T, "tab_attack.tex"), "w").write("\n".join(L))


if __name__ == "__main__":
    tab_solvers()
    if os.path.exists(os.path.join(R, "expF.json")):
        tab_attack()
    print("ok")
