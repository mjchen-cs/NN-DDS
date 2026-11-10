"""Solvers for the normalized trust-aware offloading problem (P2)."""
import time
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from model import Problem, LN2


def project_cone(pb, x):
    """Projection onto the closed convex set X = {t_u >= a b_2} (rate <= r_ext),
    keeping b_1 + b_2 = 1; this keeps the state inside the domain of E_u."""
    p = pb.p
    a = p.beta_u * p.btask / (pb.L.r_ext * p.Lm)
    b, t = x[1], x[2]
    if b > 0 and t < a * b:
        s = (b + a * t) / (1 + a * a)
        if s <= 0:
            b, t = 0.0, max(t, 0.0)
        else:
            b, t = s, a * s
        x = x.copy(); x[1], x[2], x[0] = b, t, x[0] + (x[1] - b)
    return x


# --------------------------------------------------------------------------
# NN-DDS: augmented primal-dual neural dynamics (forward-Euler discretization)
#   eps dx/dt   = -(I-P)[grad f + grad g^T (lam + c g)_+ + mu (x - xbar)] - sigma W^T sgn(Wx-b)
#   eps dlam/dt = eta [ (lam + c g)_+ - lam ] / c
# --------------------------------------------------------------------------
def nndds(pb: Problem, x0=None, lam0=None, c=5.0, eta=20.0, mu=1e-4, sigma=1.0,
          h=0.1, iters=3000, tol=1e-7, record=False, xbar=None):
    n, m = 4, pb.A.shape[0]
    W = pb.W
    Pw = W.T @ np.linalg.solve(W @ W.T, W)
    IP = np.eye(n) - Pw
    rng = pb.feasible()
    if x0 is None:
        x2 = 0.5 * (rng[0] + rng[1]) if rng else 0.5
        x0 = np.array([1 - x2, x2, 0.5, 0.05])
    x = x0.copy()
    lam = np.zeros(m) if lam0 is None else lam0.copy()
    xbar = x0.copy() if xbar is None else xbar
    hist = []
    nfev = 0
    for k in range(iters):
        fx, gf = pb.f(x); nfev += 1
        gx = pb.g(x)
        mu_plus = np.maximum(lam + c * gx, 0.0)
        gradx = gf + pb.A.T @ mu_plus + mu * (x - xbar)
        eq = W @ x - pb.bvec
        dx = -IP @ gradx - sigma * (W.T @ np.sign(np.where(np.abs(eq) < 1e-12, 0.0, eq)))
        dlam = eta * (mu_plus - lam) / c
        # adaptive step = positive time re-parametrization of the ODE (same trajectory)
        F = np.r_[dx, dlam]; z = np.r_[x, lam]
        hk = h
        if k > 0:
            dz = np.linalg.norm(z - zprev)
            if dz > 0:
                Lk = np.linalg.norm(F - Fprev) / dz
                hk = min(h, 0.5 / max(Lk, 1e-12))
        zprev, Fprev = z, F
        x = x + hk * dx
        x = project_cone(pb, x)
        lam = lam + hk * dlam
        if record:
            hist.append((k, fx, pb.viol(x), x.copy()))
        if k > 50 and np.linalg.norm(dx) < tol and np.linalg.norm(dlam) < tol:
            break
    return dict(x=x, lam=lam, it=k + 1, nfev=nfev, hist=hist)


# --------------------------------------------------------------------------
# Arrow-Hurwicz primal-dual gradient (non-augmented), projected multipliers
# --------------------------------------------------------------------------
def pdgd(pb: Problem, x0=None, h=0.05, iters=3000, tol=1e-7, record=False):
    n, m = 4, pb.A.shape[0]
    W = pb.W
    IP = np.eye(n) - W.T @ np.linalg.solve(W @ W.T, W)
    rng = pb.feasible()
    if x0 is None:
        x2 = 0.5 * (rng[0] + rng[1]) if rng else 0.5
        x0 = np.array([1 - x2, x2, 0.5, 0.05])
    x, lam = x0.copy(), np.zeros(m)
    hist = []
    for k in range(iters):
        fx, gf = pb.f(x)
        dx = -IP @ (gf + pb.A.T @ lam)
        x = project_cone(pb, x + h * dx)
        lam = np.maximum(lam + h * pb.g(x), 0.0)
        if record:
            hist.append((k, fx, pb.viol(x), x.copy()))
    return dict(x=x, it=iters, nfev=iters, hist=hist)


# --------------------------------------------------------------------------
# Interior-point (Clarabel) on the exact exponential-cone reformulation
#   tu*log2(1+lam_i e_i/tu) = -rel_entr(tu, tu+lam_i e_i)/ln2
# --------------------------------------------------------------------------
def ipm(pb: Problem, solver="CLARABEL"):
    p, L = pb.p, pb.L
    B, Lm = p.btask, p.Lm
    lam = L.lam
    x = cp.Variable(4)
    e = cp.Variable(len(lam), nonneg=True)   # radiated energy per eigen-mode [J]
    tu = x[2] * Lm
    rate_bits = cp.sum(-cp.rel_entr(tu, tu + cp.multiply(lam, e))) / LN2 * p.Wu
    Eu = p.pu * tu + p.alpha_u * (cp.sum(e) + L.P_an * tu)
    obj = (Eu + p.pd * Lm * x[3] + p.alpha_d * p.beta_d * B * x[1] + p.eps_loc * B * x[0]
           + p.rho * p.kappa * p.eps_loc * B * x[1]) / pb.Eref
    cons = [pb.A @ x <= pb.c, x[0] + x[1] == 1,
            rate_bits >= p.beta_u * B * x[1],
            cp.sum(e) + L.P_an * tu <= p.Pmax * tu]
    prob = cp.Problem(cp.Minimize(obj), cons)
    t0 = time.perf_counter()
    try:
        prob.solve(solver=solver)
    except Exception:
        prob.solve(solver="SCS", eps=1e-9, max_iters=200000)
    t1 = time.perf_counter()
    st = prob.solver_stats
    return dict(x=np.array(x.value).ravel() if x.value is not None else None,
                time=t1 - t0, solve_time=st.solve_time if st and st.solve_time else t1 - t0,
                it=st.num_iters if st else None, status=prob.status)


# --------------------------------------------------------------------------
# Structure-exploiting 1-D search (extension of Munoz et al., TVT 2015):
#   for fixed b2 the optimal rate is clip(r*, r_min(b2), r_max); F(b2) is convex
# --------------------------------------------------------------------------
def bisection(pb: Problem, tol=1e-10):
    rng = pb.feasible()
    if rng is None:
        return dict(x=None, it=0)
    rs = pb.L.r_star()
    a, b = rng
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, _ = pb.F1(c, rs); fd, _ = pb.F1(d, rs)
    it = 0
    while b - a > tol:
        it += 1
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc, _ = pb.F1(c, rs)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd, _ = pb.F1(d, rs)
    cands = [pb.F1(v, rs) for v in (0.5 * (a + b), rng[0], rng[1])]
    best = min(cands, key=lambda z: z[0])
    return dict(x=best[1], it=it)


# --------------------------------------------------------------------------
# SLSQP (sequential quadratic programming)
# --------------------------------------------------------------------------
def slsqp(pb: Problem, x0=None):
    rng = pb.feasible()
    if x0 is None:
        x2 = 0.5 * (rng[0] + rng[1]) if rng else 0.5
        x0 = np.array([1 - x2, x2, 0.5, 0.05])
    cons = [dict(type="ineq", fun=lambda x: pb.c - pb.A @ x, jac=lambda x: -pb.A),
            dict(type="eq", fun=lambda x: x[0] + x[1] - 1.0, jac=lambda x: np.array([[1., 1, 0, 0]]))]
    res = minimize(lambda x: pb.f(x)[0], x0, jac=lambda x: pb.f(x)[1], constraints=cons,
                   method="SLSQP", options=dict(maxiter=500, ftol=1e-12))
    return dict(x=res.x, it=res.nit, nfev=res.nfev)


# --------------------------------------------------------------------------
# Alternating optimization (block coordinate descent on b2 and tu)
# --------------------------------------------------------------------------
def _golden(fun, a, b, tol=1e-10):
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = fun(c), fun(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc; c = b - g * (b - a); fc = fun(c)
        else:
            a, c, fc = c, d, fd; d = a + g * (b - a); fd = fun(d)
    return 0.5 * (a + b)


def ao(pb: Problem, x0=None, outer=50):
    p, L = pb.p, pb.L
    B, Lm = p.btask, p.Lm
    rng = pb.feasible()
    if rng is None:
        return dict(x=None, it=0)
    if x0 is None:
        x2 = 0.5 * (rng[0] + rng[1])
        tu, td = pb.tu_of_b2(x2, L.r_max)       # start at maximum rate
        x0 = np.array([1 - x2, x2, tu / Lm, td / Lm])
    x = x0.copy()
    BIG = 1e6
    for k in range(outer):
        xo = x.copy()
        # block 1: b2 with tu fixed (td tied to b2)
        def fb(v):
            y = np.array([1 - v, v, x[2], p.beta_d * B * v / (L.Rd * Lm)])
            return pb.f(y)[0] + BIG * max(np.max(pb.g(y)), 0.0)
        v = _golden(fb, rng[0], rng[1])
        x = np.array([1 - v, v, x[2], p.beta_d * B * v / (L.Rd * Lm)])
        # block 2: tu with b2 fixed
        tmax = (Lm - p.tv - (p.tau2 + p.kappa * p.tau1) * B * v) / Lm - x[3]
        tlo = p.beta_u * B * v / (L.r_max * Lm)
        if tmax > tlo:
            def ft(w):
                y = x.copy(); y[2] = w
                return pb.f(y)[0]
            x[2] = _golden(ft, tlo, tmax)
        if np.linalg.norm(x - xo) < 1e-10:
            break
    return dict(x=x, it=k + 1)


# --------------------------------------------------------------------------
# Particle swarm optimization (penalty), fixed evaluation budget
# --------------------------------------------------------------------------
def pso(pb: Problem, budget=4000, npop=40, seed=0):
    p, L = pb.p, pb.L
    B, Lm = p.btask, p.Lm
    rs = np.random.default_rng(seed)
    rng = pb.feasible()
    lo = np.array([rng[0] if rng else 0.0, 1e-4]); hi = np.array([1.0, 1.0])

    def full(z):  # z = (x2, tu/Lm); td at its lower bound
        x2 = z[0]
        return np.array([1 - x2, x2, z[1], p.beta_d * B * x2 / (L.Rd * Lm)])

    def J(z):
        x = full(z)
        return pb.f(x)[0] + 1e3 * max(np.max(pb.g(x)), 0.0)

    X = lo + (hi - lo) * rs.random((npop, 2)); V = np.zeros_like(X)
    Pb = X.copy(); Pf = np.array([J(z) for z in X]); nfev = npop
    gi = np.argmin(Pf)
    while nfev + npop <= budget:
        r1, r2 = rs.random((npop, 2)), rs.random((npop, 2))
        V = 0.7 * V + 1.5 * r1 * (Pb - X) + 1.5 * r2 * (Pb[gi] - X)
        X = np.clip(X + V, lo, hi)
        F = np.array([J(z) for z in X]); nfev += npop
        imp = F < Pf; Pb[imp] = X[imp]; Pf[imp] = F[imp]
        gi = np.argmin(Pf)
    return dict(x=full(Pb[gi]), it=nfev // npop, nfev=nfev)
