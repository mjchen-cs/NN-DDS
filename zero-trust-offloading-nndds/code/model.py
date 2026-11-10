"""
System model for trust-aware partial offloading over MIMO links.

All quantities are in SI units unless stated otherwise.
Channel matrices are normalized such that the noise power is 1, i.e.
H = sqrt(gamma_bar) * G with G having i.i.d. CN(0,1) entries and
gamma_bar the mean channel gain normalized by the noise power (1/W).
"""
import numpy as np
from dataclasses import dataclass, field, replace

LN2 = np.log(2.0)


@dataclass
class Params:
    # radio
    Wu: float = 20e6          # uplink bandwidth [Hz]
    Wd: float = 20e6          # downlink bandwidth [Hz]
    pu: float = 0.3           # uplink circuit power [W]
    alpha_u: float = 4.0      # inverse PA efficiency
    pd: float = 0.2           # downlink circuit power [W]
    alpha_d: float = 2.5e-9   # decoding energy [J/bit] (=2.5 mW/Mbps)
    Pmax: float = 0.2         # device max radiated power [W] (23 dBm)
    Pa: float = 1.0           # AP max radiated power [W]
    eta_cap: float = 6.0      # spectral-efficiency cap per stream [bit/s/Hz]
    # computing
    tau1: float = 1e-8        # local time per bit [s/bit]
    tau2: float = 1e-9        # edge time per bit [s/bit]
    eps_loc: float = 1.2e-7   # local energy per bit [J/bit]
    btask: float = 4e6        # task size [bit]
    beta_u: float = 1.0       # uplink overhead factor
    beta_d: float = 0.1       # downlink (result) size factor
    Lm: float = 0.05          # deadline [s]
    # shield
    kappa: float = 0.5        # relative cost of the lightweight fallback model
    tv: float = 0.5e-3        # verification time [s]
    rho: float = 0.0          # fallback probability used by the optimizer
    phi_an: float = 0.0       # fraction of Pmax spent on artificial noise
    # numerics
    tmin: float = 1e-6


# --------------------------------------------------------------------------
# channels
# --------------------------------------------------------------------------
def rayleigh(nr, nt, rng, gain=1.0):
    return np.sqrt(gain / 2) * (rng.standard_normal((nr, nt)) + 1j * rng.standard_normal((nr, nt)))


def eig_gains(H, R=None):
    """Eigenvalues (desc) of H^H R^{-1} H (interference-whitened channel)."""
    if R is None:
        M = H.conj().T @ H
    else:
        M = H.conj().T @ np.linalg.solve(R, H)
    lam = np.linalg.eigvalsh(M)[::-1]
    lam = lam[lam > 1e-12 * max(lam.max(), 1e-30)]
    return np.maximum(lam.real, 1e-30)


# --------------------------------------------------------------------------
# water-filling
# --------------------------------------------------------------------------
def wf_min_power(R, lam):
    """Min total power to support spectral efficiency R [bit/s/Hz] over
    eigen-modes lam (desc).  Returns (P, S, G)."""
    n = len(lam)
    G = 1
    S = 2.0 ** R / lam[0]
    for k in range(1, n + 1):
        Sk = 2.0 ** (R / k) / np.exp(np.mean(np.log(lam[:k])))
        if Sk > 1.0 / lam[k - 1]:
            G, S = k, Sk
    P = np.sum(S - 1.0 / lam[:G])
    return max(P, 0.0), S, G


def wf_capacity(P, lam):
    """Water-filling capacity [bit/s/Hz] at total power P."""
    n = len(lam)
    for G in range(n, 0, -1):
        mu = (P + np.sum(1.0 / lam[:G])) / G
        if mu > 1.0 / lam[G - 1]:
            return np.sum(np.log2(mu * lam[:G]))
    return 0.0


class Link:
    """Pre-computes everything that depends on one channel realization."""

    def __init__(self, p: Params, lam_u, lam_d, n_an_dim=0):
        self.p = p
        self.lam = np.asarray(lam_u)
        self.n_streams = len(self.lam)
        self.P_an = p.phi_an * p.Pmax if n_an_dim > 0 else 0.0
        Pav = p.Pmax - self.P_an
        cap_u = wf_capacity(Pav, self.lam)
        self.r_max = p.Wu * min(cap_u, p.eta_cap * self.n_streams)      # [bit/s]
        cap_d = wf_capacity(p.Pa, np.asarray(lam_d))
        self.Rd = p.Wd * min(cap_d, p.eta_cap * len(lam_d))             # [bit/s]
        # quadratic extension point for numerical safety (convex extension)
        self.r_ext = 1.5 * self.r_max
        P0, S0, G0 = wf_min_power(self.r_ext / p.Wu, self.lam)
        self.ext = (P0, S0 * LN2 / p.Wu, S0 * LN2 ** 2 / (G0 * p.Wu ** 2))

    # power P(r) [W] and its derivatives in r [bit/s]
    def P(self, r):
        p = self.p
        if r <= self.r_ext:
            P, S, G = wf_min_power(r / p.Wu, self.lam)
            return P, S * LN2 / p.Wu, S * LN2 ** 2 / (G * p.Wu ** 2)
        P0, d1, d2 = self.ext
        dr = r - self.r_ext
        return P0 + d1 * dr + 0.5 * d2 * dr * dr, d1 + d2 * dr, d2

    # uplink energy E_u(b,t) = t [pu + alpha_u (P(b/t) + P_an)] and gradient
    def Eu(self, b, t):
        p = self.p
        t = max(t, p.tmin)
        if b <= 0:   # convex C^1 linear extension of P(r) for r <= 0
            d0 = LN2 / (p.Wu * self.lam[0])
            E = t * (p.pu + p.alpha_u * self.P_an) + p.alpha_u * d0 * b
            return E, p.alpha_u * d0, p.pu + p.alpha_u * self.P_an
        r = b / t
        P, dP, _ = self.P(r)
        E = t * (p.pu + p.alpha_u * (P + self.P_an))
        dEdb = p.alpha_u * dP
        dEdt = p.pu + p.alpha_u * (P + self.P_an) - p.alpha_u * r * dP
        return E, dEdb, dEdt

    def Ebar(self, r):
        """Energy per bit as a function of the rate."""
        p = self.p
        P, _, _ = self.P(r)
        return (p.pu + p.alpha_u * (P + self.P_an)) / r

    def r_star(self):
        """argmin_r Ebar(r) on (0, r_max] (quasiconvex => golden section)."""
        a, b = 1e3, self.r_max
        g = (np.sqrt(5) - 1) / 2
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = self.Ebar(c), self.Ebar(d)
        for _ in range(100):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a); fc = self.Ebar(c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a); fd = self.Ebar(d)
            if b - a < 1e-7 * self.r_max:
                break
        return 0.5 * (a + b)


# --------------------------------------------------------------------------
# normalized problem (P2):   x = [b1/btask, b2/btask, tu/Lm, td/Lm]
#   min f(x)   s.t.  g(x) <= 0,  W x = 1
# --------------------------------------------------------------------------
class Problem:
    def __init__(self, link: Link):
        self.L = link
        p = link.p
        self.p = p
        self.Eref = p.eps_loc * p.btask          # local-only energy
        self.W = np.array([[1.0, 1.0, 0.0, 0.0]])
        self.bvec = np.array([1.0])
        B, Lm = p.btask, p.Lm
        a = p.tau1 * B / Lm
        # linear constraints  A x <= c   (rows g1..g6, g8); g7 is the domain t_u>=tmin
        self.A = np.array([
            [-1, 0, 0, 0],                                   # g1: b1 >= 0
            [0, -1, 0, 0],                                   # g2: b2 >= 0
            [a, p.kappa * a, 0, 0],                          # g3: fallback feasible
            [0, (p.tau2 + p.kappa * p.tau1) * B / Lm, 1, 1], # g4: safe window
            [0, p.beta_u * B / (link.r_max * Lm), -1, 0],    # g5: rate/power cap
            [0, p.beta_d * B / (link.Rd * Lm), 0, -1],       # g6: downlink rate
            [0, 0, -1, 0],                                   # g7: t_u >= 0
            [0, 0, 0, -1],                                   # g8: t_d >= 0
        ], dtype=float)
        self.c = np.array([0, 0, 1, 1 - p.tv / Lm, 0, 0, 0, 0], dtype=float)
        # row normalization (does not change the feasible set; improves conditioning)
        nrm = np.linalg.norm(self.A, axis=1)
        self.A /= nrm[:, None]
        self.c /= nrm

    def f(self, x):
        p, L = self.p, self.L
        B, Lm = p.btask, p.Lm
        b2 = max(x[1], 0.0) * B * p.beta_u
        Eu, dEb, dEt = L.Eu(b2, max(x[2], p.tmin / Lm) * Lm)
        val = (Eu + p.pd * Lm * x[3] + p.alpha_d * p.beta_d * B * x[1]
               + p.eps_loc * B * x[0] + p.rho * p.kappa * p.eps_loc * B * x[1])
        grad = np.array([
            p.eps_loc * B,
            dEb * B * p.beta_u + p.alpha_d * p.beta_d * B + p.rho * p.kappa * p.eps_loc * B,
            dEt * Lm,
            p.pd * Lm,
        ])
        return val / self.Eref, grad / self.Eref

    def g(self, x):
        return self.A @ x - self.c

    def viol(self, x):
        return max(np.max(self.g(x)), 0.0) + abs(x[0] + x[1] - 1.0)

    def energy(self, x):
        return self.f(x)[0] * self.Eref

    def feasible(self):
        """Feasibility check of (P2); returns (lo, hi) range of b2 fraction."""
        p, L = self.p, self.L
        B, Lm = p.btask, p.Lm
        lo = 0.0
        if p.tau1 * B > Lm:
            if p.kappa >= 1:
                return None
            lo = (p.tau1 * B - Lm) / (p.tau1 * B * (1 - p.kappa))
        # tu >= beta_u B x2 / r_max and tu <= Lm - tv - (tau2+kappa tau1)B x2 - td
        den = (p.beta_u / L.r_max + p.tau2 + p.kappa * p.tau1 + p.beta_d / L.Rd) * B
        hi = min(1.0, (Lm - p.tv) / den)
        if hi < lo:
            return None
        return lo, hi

    # --- structure-exploiting exact solution (Munoz et al. style) ------------
    def tu_of_b2(self, x2, rs):
        p, L = self.p, self.L
        B, Lm = p.btask, p.Lm
        bu = p.beta_u * B * x2
        td = p.beta_d * B * x2 / L.Rd
        tmax = Lm - p.tv - (p.tau2 + p.kappa * p.tau1) * B * x2 - td
        tlo = bu / L.r_max
        t = bu / rs if x2 > 0 else 0.0
        return min(max(t, tlo), tmax), td

    def F1(self, x2, rs):
        tu, td = self.tu_of_b2(x2, rs)
        x = np.array([1 - x2, x2, tu / self.p.Lm, td / self.p.Lm])
        return self.f(x)[0], x
