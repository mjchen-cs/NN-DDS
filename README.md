# NN-DDS: Zero-Trust Offloading with Signal-Level Protection for MIMO 6G Edge Networks

Repository: <https://github.com/mjchen-cs/NN-DDS>

Code and data for the paper

> M. Chen, H. Qin, H. Shi, M. Yi, J. Liang, S. Mumtaz, A. Farouk,
> "Zero-Trust Offloading with Signal-Level Protection for MIMO 6G Edge Networks: A Neural Differential Dynamic Approach,"
> submitted to *IEEE Journal of Selected Topics in Signal Processing*, Special Issue on Signal Processing for Trustworthiness in 6G Wireless Ecosystems.

The repository contains:

* a MIMO partial-offloading model with minimum-power water-filling, null-space artificial noise and the trust-aware problem (P2);
* the **NN-DDS** solver (augmented primal–dual neural dynamics) and all baselines: interior-point (Clarabel/CVXPY), structure-exploiting bisection, SLSQP, alternating optimization, primal–dual gradient dynamics, PSO and a learning-to-optimize MLP;
* the signal-level shield: a pilot-residual jamming detector, channel-based PHY-layer authentication, a command consistency test, and the safe waiting window with fallback;
* every experiment of the paper, the **summary results** (`code/data/summary/*.json`), the **raw per-instance / per-round data** (`code/data/raw/*.csv`), the figures (`figs/`) and the LaTeX tables (`tables/`).

## Layout

```
code/
  model.py          system model, water-filling, problem (P2)
  solvers.py        NN-DDS and baseline solvers
  security.py       detectors, PHY authentication, AN, IRC
  rawlog.py         helper that writes raw data to data/raw/*.csv
  exp_solvers.py    EXP-A solver comparison (Table III), EXP-B convergence (Figs. 2-3)
  exp_system.py     EXP-C tracking (Fig. 4), EXP-D system behaviour (Figs. 5, 6a)
  exp_rho.py        effect of the fallback probability (Fig. 6b)
  exp_security.py   EXP-E ROC (Fig. 7), EXP-F attacks (Table IV, Fig. 8), EXP-G AN (Fig. 9)
  exp_k025.py       Table IV rows with kappa = 0.25
  plot.py           figures  -> figs/
  tables.py         tables   -> tables/
  data/summary/     aggregated results (JSON) used by plot.py / tables.py
  data/raw/         raw data (CSV), one row per instance / round / trial
figs/               figures of the paper (PDF)
tables/             LaTeX tables of the paper
run_all.sh          reproduces everything
```

## Requirements and usage

Python ≥ 3.10:

```bash
git clone https://github.com/mjchen-cs/NN-DDS.git
cd NN-DDS
pip install -r requirements.txt
./run_all.sh            # full reproduction (about 1.5-2 h on one CPU core per chain)
# or individual steps, from the code/ folder:
(cd code && python3 exp_solvers.py && python3 plot.py && python3 tables.py)
```

All random seeds are fixed, so a re-run reproduces the summary results exactly. Only the wall-clock times in Table III depend on the machine. The published data were produced with Python 3.13, NumPy 2.5, SciPy 1.18, CVXPY 1.9 (Clarabel 0.11), scikit-learn 1.9 and Matplotlib 3.11.

To regenerate figures and tables from the stored data without re-running the experiments, run `python3 plot.py` and `python3 tables.py` in `code/`.

## Raw data dictionary (`code/data/raw/`)

All energies are device energies. `*_norm` values are normalized by the local-only energy `eps_loc * b_task` (0.48 J).

| file | one row per | main columns |
|---|---|---|
| `expA_solvers.csv` | instance × solver (4 configs × 200 instances × 8 solvers) | `config, instance, gain_dB, solver, energy_norm, energy_ipm_norm, rel_gap, violation, time_s, iterations, b2_frac, tu_s, td_s` |
| `expB_convergence.csv` | iteration of NN-DDS (c=5), NN-DDS (c=1), PDGD | `method, iteration, rel_gap, violation, b2_frac, tu_ms` |
| `expC_tracking.csv` | channel update (1 ms) × Doppler × K | `fD_Hz, K, update, energy_opt_norm, gap_warm, gap_cold, viol_warm_raw` |
| `expD_gain.csv` | channel realization × gain × config | `config, gain_dB, energy_J, saving, b2_frac, latency_ms, rate_Mbps, tu_s, td_s` |
| `expD_deadline.csv` | channel × deadline × kappa (4×2, 20 dB) | `kappa, Lm_s, energy_J, saving, b2_frac` |
| `expRho.csv` | channel × rho × kappa × gain (4×2) | `kappa, gain_dB, rho, energy_J, b2_frac` |
| `expE_jam_stats.csv` | Monte-Carlo trial of the jamming test | `Nr, trial, T_H0, T_H1` (JNR = 0 dB, L_p = 16) |
| `expE_auth_stats.csv` | Monte-Carlo trial of PHY authentication | `K, trial, T_legit, T_spoof` (f_D = 50 Hz, Δt = 5 ms) |
| `expF_rounds.csv` | offloading round × scheme × config (π = 0.1) | `config, scheme, round, ev_persist, ev_react, ev_spoof, rho_hat, energy_J, deadline_violation, attack_success, intercepted, fallback, b2_frac, jam_detected` |
| `expF_sweep_rounds.csv` | round × scheme × attack probability (4×2) | same columns as `expF_rounds.csv` |
| `expF_k025_rounds.csv` | round × config, proposed scheme with κ = 0.25 | same columns |
| `expG_rounds.csv` | round × AN fraction φ × config (no active attack) | same columns, plus `phi_an` |

Schemes in `expF_*`: `local` (local only), `naive` (security-agnostic offloading), `decoupled` (shield attached to the security-agnostic plan), `proposed` (shield plus trust-aware problem P2). The detector thresholds of the jamming and authentication tests follow from their closed-form null distributions, `2T ~ chi2(2·N_r·L_p)` and `2T ~ chi2(2K)`, so ROC points can be recomputed directly from `expE_*.csv`.

## Citation

Please cite the paper above if you use this code or data. A BibTeX entry will be added on publication.

## License

Released under the [MIT License](LICENSE). The data in `code/data/` are released under the same terms.
