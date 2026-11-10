"""Signal-level components of the trustworthy offloading shield.

  * pilot-residual energy detector for jamming on the response frame
  * channel-based PHY-layer authentication of the response frame (anti-spoofing)
  * consistency (semantic) check of the returned command against the local
    lightweight model, with projection onto the admissible set
  * artificial noise (AN) in the null space of the legitimate uplink channel
  * interference-whitened (IRC) reception at the AP against jamming
"""
import numpy as np
from scipy.stats import chi2, ncx2
from scipy.special import j0
from model import rayleigh, LN2


# --------------------------------------------------------------------------
# 1) jamming detector: T = ||Y - H_hat X||_F^2 / s0 ;  2T ~ chi2(2 N L) under H0
# --------------------------------------------------------------------------
def jam_threshold(N, Lp, pfa):
    return 0.5 * chi2.isf(pfa, 2 * N * Lp)


def jam_pd(N, Lp, pfa, jnr, s0=1.0):
    """Closed-form detection probability: under H1 the residual variance is s0+jnr."""
    eta = jam_threshold(N, Lp, pfa)
    return chi2.sf(2 * eta * s0 / (s0 + jnr), 2 * N * Lp)


def jam_stat_mc(N, Lp, jnr, rng, s0=1.0):
    e = np.sqrt((s0 + jnr) / 2) * (rng.standard_normal((N, Lp)) + 1j * rng.standard_normal((N, Lp)))
    return np.sum(np.abs(e) ** 2) / s0


# --------------------------------------------------------------------------
# 2) PHY authentication: D = H_hat_resp - H_hat_ref,  T = ||D||^2 / s0
#    legit:  per-entry var s0 = 2(1-rho_t) + 2 se2,  spoof: s1 = 1 + gs + 2 se2
# --------------------------------------------------------------------------
def auth_params(K, fD, dt, se2, gs=1.0):
    rho_t = j0(2 * np.pi * fD * dt)
    s0 = 2 * (1 - rho_t) + 2 * se2
    s1 = 1 + gs + 2 * se2
    return s0, s1


def auth_threshold(K, pfa):
    return 0.5 * chi2.isf(pfa, 2 * K)


def auth_pd(K, pfa, s0, s1):
    eta = auth_threshold(K, pfa)
    return chi2.sf(2 * eta * s0 / s1, 2 * K)


def auth_stat_mc(K, legit, fD, dt, se2, rng, gs=1.0):
    """Monte-Carlo with an explicit Gauss-Markov channel (unit-variance entries)."""
    rho_t = j0(2 * np.pi * fD * dt)
    Href = rayleigh(K, 1, rng).ravel()
    if legit:
        Hr = rho_t * Href + np.sqrt(max(1 - rho_t ** 2, 0)) * rayleigh(K, 1, rng).ravel()
    else:
        Hr = np.sqrt(gs) * rayleigh(K, 1, rng).ravel()
    D = (Hr + np.sqrt(se2) * rayleigh(K, 1, rng).ravel()) - (Href + np.sqrt(se2) * rayleigh(K, 1, rng).ravel())
    s0, _ = auth_params(K, fD, dt, se2, gs)
    return np.sum(np.abs(D) ** 2) / s0


# --------------------------------------------------------------------------
# 3) consistency check of the command u (dimension d) against local model
#    u_loc = u* + n,  n ~ N(0, sl2 I).  Accept iff ||u - u_loc||^2 / sl2 <= eta_c
# --------------------------------------------------------------------------
def cons_threshold(d, pfa):
    return chi2.isf(pfa, d)


def cons_pd(d, pfa, offset2_over_sl2):
    return ncx2.sf(cons_threshold(d, pfa), d, offset2_over_sl2)


# --------------------------------------------------------------------------
# 4) eavesdropper rate with AN in the null space of H (Goel & Negi)
# --------------------------------------------------------------------------
def tx_covariance(H, P_sig, P_an):
    """Water-filling signal covariance on eig-modes of H^H H plus isotropic AN in null(H)."""
    nr, nt = H.shape
    U, s, Vh = np.linalg.svd(H)
    V = Vh.conj().T
    r = int(np.sum(s > 1e-9))
    lam = s[:r] ** 2
    # water-filling at power P_sig
    for G in range(r, 0, -1):
        mu = (P_sig + np.sum(1 / lam[:G])) / G
        if mu > 1 / lam[G - 1]:
            break
    pw = np.maximum(mu - 1 / lam, 0)
    Vs = V[:, :r]
    Q = (Vs * pw) @ Vs.conj().T
    Qan = np.zeros((nt, nt), complex)
    if nt > r and P_an > 0:
        Vn = V[:, r:]
        Qan = Vn @ Vn.conj().T * (P_an / (nt - r))
    return Q, Qan


def eve_rate(Ge, Q, Qan):
    """Eve's achievable spectral efficiency (bit/s/Hz), noise normalized to 1."""
    ne = Ge.shape[0]
    Rn = np.eye(ne) + Ge @ Qan @ Ge.conj().T
    S = Ge @ Q @ Ge.conj().T
    M = np.eye(ne) + np.linalg.solve(Rn, S)
    return max(np.real(np.log2(np.linalg.det(M))), 0.0)


def bob_rate_irc(H, Q, jnr_lin, hJ):
    """AP spectral efficiency with interference-rejection combining (whitening)."""
    nr = H.shape[0]
    R = np.eye(nr) + jnr_lin * np.outer(hJ, hJ.conj())
    M = np.eye(nr) + np.linalg.solve(R, H @ Q @ H.conj().T)
    return max(np.real(np.log2(np.linalg.det(M))), 0.0)
