"""
QTEG_JEL_Arctic.py  (v4 — full covariance delta method for NA survival CI)
===========================================================================
Arctic HPC simulation for Paper 2: JEL inference for QTEG.

Statistical core ported directly from QTEG_JEL_Analysis_v2_fix.py:
  - qteg_mle        : full L-BFGS-B optimisation with Hessian SEs + cov_ab
  - compute_pseudovalues : single LOO pass, stores loo_alpha/loo_beta
  - _solve_lambda        : Newton-Raphson Lagrange solver   [Eq. 22]
  - jel_ratio / ajel_ratio / ejel_ratio                    [Eq. 21,25,29]
  - _ci_from_ratio       : grid inversion, 1000-point search
  - jel_ci / ajel_ci / ejel_ci                             [Eq. 32-34]
  - survival_pseudovalues_from_pv : reuses stored LOO MLEs [Eq. 39]
  - survival CI          : grid search over (0,1)           [Eq. 41-43]
  - na_survival_ci       : logit-transformed Wald CI for S(t0)
                           uses full 2x2 Hessian covariance (cov_ab)
  - ajel_ratio           : BAEL — Emerson & Owen (2009) two-obs version

v4 changes vs v3:
  - qteg_mle now returns cov_ab = Cov(alpha_hat, beta_hat) from Hessian
  - na_survival_ci now accepts cov_ab and uses the full delta-method formula:
      Var(S(t0)) = (dS/da)^2 Var(a) + (dS/db)^2 Var(b) + 2(dS/da)(dS/db)Cov(a,b)
  - run_block and run_real_data pass cov_ab through to na_survival_ci
  - Resume-from-checkpoint logic retained from v3

Arctic structure:
  - 36-block SLURM array
  - partial checkpoint every 50 reps (with _raw_counts for resume)
  - per-block JSON output
  - merged JSON + CSV
  - --test / --block / --n_sim / --merge / --realdata flags

Authors: Taiwo Michael Ayeni and Yichuan Zhao
Georgia State University, 2026
"""

import numpy as np
from scipy.special import (gammaln, digamma, gammainc,
                            gammaincc, gammaincinv)
from scipy.optimize import minimize, brentq
from scipy.stats   import chi2
import json, os, argparse, time, csv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)

# ── Simulation parameters ─────────────────────────────────────────────
N_SIM_DEFAULT    = 5000
BASE_SEED        = 20260415
CHECKPOINT_EVERY = 50

NULL_SCENARIOS = [
    (1.5, 0.5),
    (2.0, 1.0),
    (3.0, 2.0),
]
NOMINALS     = [0.90, 0.95, 0.99]
SAMPLE_SIZES = [30, 50, 100, 200]

# t0 = median of QTEG under each null scenario
T0_VALS = {
    (1.5, 0.5): float((gammaincinv(1.5, 0.5) / 0.5)**2),
    (2.0, 1.0): float((gammaincinv(2.0, 0.5) / 1.0)**2),
    (3.0, 2.0): float((gammaincinv(3.0, 0.5) / 2.0)**2),
}

# ── QTEG functions ────────────────────────────────────────────────────

def qteg_logpdf(y, alpha, beta):
    return (alpha*np.log(beta) - np.log(2.0) - gammaln(alpha)
            + ((alpha-2)/2.0)*np.log(y) - beta*np.sqrt(y))

def qteg_sf(y, alpha, beta):
    """Survival function S(y) = 1 - CDF(y)."""
    return gammaincc(alpha, beta * np.sqrt(np.maximum(y, 1e-300)))

def qteg_sample(alpha, beta, n, rng):
    return rng.gamma(alpha, 1.0/beta, n) ** 2

# ── MLE (full L-BFGS-B with Hessian SEs and covariance) ──────────────

def qteg_mle(y):
    """
    MLE for QTEG(alpha, beta).
    Multi-start L-BFGS-B optimisation.  [Eq. 23-26 of paper]

    Returns dict with keys:
      alpha, beta, se_alpha, se_beta, cov_ab, logL, n
    cov_ab = Cov(alpha_hat, beta_hat) from the full 2x2 Hessian inverse.
    Returns None on failure.
    """
    y  = np.asarray(y, float)
    sy = np.mean(np.sqrt(y))

    def neg_ll(params):
        a, b = params
        if a <= 0 or b <= 0: return np.inf
        return -np.sum(qteg_logpdf(y, a, b))

    best_res, best_val = None, np.inf
    for a0 in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]:
        b0 = a0 / sy
        try:
            res = minimize(neg_ll, [a0, b0], method='L-BFGS-B',
                           bounds=[(1e-6, None), (1e-6, None)],
                           options={'ftol': 1e-13, 'gtol': 1e-9,
                                    'maxiter': 5000})
            if res.fun < best_val:
                best_val, best_res = res.fun, res
        except Exception:
            pass

    if best_res is None:
        return None

    ah, bh = best_res.x
    logL   = -best_val

    # Full 2x2 Hessian — retains off-diagonal for delta method
    h = 1e-5
    try:
        H = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                ei = np.zeros(2); ei[i] = h
                ej = np.zeros(2); ej[j] = h
                H[i,j] = (neg_ll(best_res.x+ei+ej) - neg_ll(best_res.x+ei-ej)
                          -neg_ll(best_res.x-ei+ej) + neg_ll(best_res.x-ei-ej)
                          ) / (4*h**2)
        cov    = np.linalg.inv(H)
        se_a   = float(np.sqrt(max(cov[0,0], 0.0)))
        se_b   = float(np.sqrt(max(cov[1,1], 0.0)))
        cov_ab = float(cov[0,1])          # Cov(alpha_hat, beta_hat)
    except Exception:
        se_a = se_b = np.nan
        cov_ab = 0.0

    return dict(alpha=ah, beta=bh, se_alpha=se_a, se_beta=se_b,
                cov_ab=cov_ab, logL=logL, n=len(y))

# ── Pseudo-values (single LOO pass) ──────────────────────────────────

def compute_pseudovalues(y):
    """
    Jackknife pseudo-values for alpha and beta.  [Eq. 14-15]
    Stores loo_alpha and loo_beta for reuse in survival pseudo-values.
    Returns dict or None.
    """
    y    = np.asarray(y, float)
    n    = len(y)
    fit0 = qteg_mle(y)
    if fit0 is None:
        return None

    ah = fit0['alpha']
    bh = fit0['beta']

    pv_alpha  = np.empty(n)
    pv_beta   = np.empty(n)
    loo_alpha = np.empty(n)
    loo_beta  = np.empty(n)

    for i in range(n):
        y_loo = np.delete(y, i)
        fit_i = qteg_mle(y_loo)
        if fit_i is None:
            loo_alpha[i] = ah
            loo_beta[i]  = bh
        else:
            loo_alpha[i] = fit_i['alpha']
            loo_beta[i]  = fit_i['beta']
        pv_alpha[i] = n*ah - (n-1)*loo_alpha[i]   # Eq. 14
        pv_beta[i]  = n*bh - (n-1)*loo_beta[i]    # Eq. 15

    return dict(alpha=ah, beta=bh, se_alpha=fit0['se_alpha'],
                se_beta=fit0['se_beta'], cov_ab=fit0['cov_ab'],
                pv_alpha=pv_alpha, pv_beta=pv_beta,
                loo_alpha=loo_alpha, loo_beta=loo_beta,
                n=n, mle=fit0)

# ── Lagrange solver and ratio functions ───────────────────────────────

def _solve_lambda(V, theta0, max_iter=200, tol=1e-12):
    """Newton-Raphson solver for JEL constraint.  [Eq. 22]"""
    W = V - theta0
    n = len(W)

    if theta0 <= np.min(V) or theta0 >= np.max(V):
        raise ValueError("theta0 outside convex hull of pseudo-values")

    pos_W  = W[W > 0]
    neg_W  = W[W < 0]
    lam_lo = (-1.0/pos_W.min()) if len(pos_W) > 0 else -1e8
    lam_hi = (-1.0/neg_W.max()) if len(neg_W) > 0 else  1e8
    eps    = 1e-10
    lam_lo += eps
    lam_hi -= eps

    lam = 0.0
    for _ in range(max_iter):
        denom = 1.0 + lam * W
        if np.any(denom <= 0):
            lam *= 0.5
            continue
        g  = np.sum(W / denom) / n
        dg = -np.sum((W / denom)**2) / n
        if abs(dg) < 1e-14:
            break
        lam_new = lam - g/dg
        lam = float(np.clip(lam_new, lam_lo, lam_hi))
        if abs(g/dg) < tol:
            break
    return lam


def jel_ratio(V, theta0):
    """JEL log-likelihood ratio.  [Eq. 21]"""
    try:
        lam = _solve_lambda(V, theta0)
        W   = V - theta0
        return float(2.0 * np.sum(np.log(1.0 + lam * W)))
    except (ValueError, RuntimeError):
        return np.inf


def ajel_ratio(V, theta0):
    """
    BAEL log-likelihood ratio following Emerson & Owen (2009).
    Adds TWO symmetric artificial observations to expand the convex
    hull in both directions.

    Artificial observations:
      g_{n+1} = theta0 + a_n * sigma_hat   (upper)
      g_{n+2} = theta0 - a_n * sigma_hat   (lower)

    where a_n = max(1, log(n)/2) [Chen 2008] and
    sigma_hat = std(V) is the scale of the pseudo-values.
    """
    n       = len(V)
    a_n     = max(1.0, np.log(n) / 2.0)
    sigma   = np.std(V, ddof=1)
    V_aug   = np.append(V, [theta0 + a_n * sigma,
                             theta0 - a_n * sigma])
    W_aug   = V_aug - theta0
    try:
        lam = _solve_lambda(V_aug, theta0)
        return float(2.0 * np.sum(np.log(1.0 + lam * W_aug)))
    except (ValueError, RuntimeError):
        return np.inf


def ejel_ratio(V, theta0, alpha_hat):
    """EJEL log-likelihood ratio.  [Eq. 29-31]"""
    def hcn(t):
        ell = jel_ratio(V, t) if (np.min(V) < t < np.max(V)) else np.inf
        if np.isinf(ell):
            return np.inf
        gamma = 1.0 + ell / (2.0 * len(V))
        return alpha_hat + gamma * (t - alpha_hat)

    lo = np.min(V) + 1e-8
    hi = np.max(V) - 1e-8
    try:
        t_inv = brentq(lambda t: hcn(t) - theta0, lo, hi,
                       xtol=1e-10, maxiter=200)
        return jel_ratio(V, t_inv)
    except (ValueError, RuntimeError):
        return np.inf

# ── CI from ratio (grid inversion) ───────────────────────────────────

def _ci_from_ratio(ratio_fn, V, nominal, theta_hat,
                   search_half_width=None):
    """Grid inversion of any EL ratio statistic.  [Eq. 32-34]"""
    cutoff = chi2.ppf(nominal, df=1)
    if search_half_width is None:
        sd = np.std(V, ddof=1)
        search_half_width = max(3.0*sd, abs(theta_hat)*0.5 + 0.1)

    for expand in [1, 2, 4]:
        hw   = search_half_width * expand
        grid = np.linspace(max(theta_hat - hw, 1e-6),
                           theta_hat + hw, 1000)
        vals   = np.array([ratio_fn(V, t) for t in grid])
        inside = grid[vals <= cutoff]
        if len(inside) >= 2:
            return float(inside[0]), float(inside[-1])

    return (np.nan, np.nan)


def na_ci(theta_hat, se, nominal):
    """Normal approximation (Wald) CI."""
    z = np.sqrt(chi2.ppf(nominal, df=1))
    return (theta_hat - z*se, theta_hat + z*se)

def jel_ci(V, nominal, theta_hat):
    return _ci_from_ratio(jel_ratio, V, nominal, theta_hat)

def ajel_ci(V, nominal, theta_hat):
    return _ci_from_ratio(ajel_ratio, V, nominal, theta_hat)

def ejel_ci(V, nominal, theta_hat):
    ratio_fn = lambda v, t: ejel_ratio(v, t, theta_hat)
    return _ci_from_ratio(ratio_fn, V, nominal, theta_hat)

# ── Survival pseudo-values ────────────────────────────────────────────

def na_survival_ci(ah, bh, se_a, se_b, t0, nominal, cov_ab=0.0):
    """
    Wald (NA) CI for S(t0) via logit-transformed delta method.

    Uses the FULL 2x2 covariance matrix including Cov(alpha_hat, beta_hat):
      Var(S(t0)) = (dS/da)^2 Var(a)
                 + (dS/db)^2 Var(b)
                 + 2 (dS/da)(dS/db) Cov(a,b)        [full delta method]

    Ignoring cov_ab (as in v3) underestimates se_psi when alpha and beta
    are positively correlated in the MLE, producing over-wide CIs and
    inflated coverage for S(t0).

    Logit transformation guarantees CI bounds remain in (0,1).
    Reference: Meeker & Escobar (1998), Statistical Methods for
               Reliability Data, Wiley.
    """
    h = 1e-5
    dS_da   = (qteg_sf(t0, ah+h, bh) - qteg_sf(t0, ah-h, bh)) / (2*h)
    dS_db   = (qteg_sf(t0, ah, bh+h) - qteg_sf(t0, ah, bh-h)) / (2*h)

    # Full delta method variance — includes off-diagonal covariance term
    var_psi = ((dS_da * se_a)**2
               + (dS_db * se_b)**2
               + 2.0 * dS_da * dS_db * cov_ab)
    se_psi  = float(np.sqrt(max(var_psi, 0.0)))

    psi_hat = float(qteg_sf(t0, ah, bh))
    z       = float(np.sqrt(chi2.ppf(nominal, df=1)))

    # Logit-scale transformation — guarantees CI in (0,1)
    psi_c    = float(np.clip(psi_hat, 1e-6, 1.0 - 1e-6))
    logit    = np.log(psi_c / (1.0 - psi_c))
    se_logit = se_psi / (psi_c * (1.0 - psi_c))
    lo = 1.0 / (1.0 + np.exp(-(logit - z * se_logit)))
    hi = 1.0 / (1.0 + np.exp(-(logit + z * se_logit)))
    return (float(lo), float(hi))


def survival_pseudovalues_from_pv(pv_dict, t0):
    """
    Survival pseudo-values reusing LOO MLEs already in pv_dict.
    Avoids a second full LOO pass.  [Eq. 39]
    """
    ah        = pv_dict['alpha']
    bh        = pv_dict['beta']
    loo_alpha = pv_dict['loo_alpha']
    loo_beta  = pv_dict['loo_beta']
    n         = pv_dict['n']

    psi_hat = float(qteg_sf(t0, ah, bh))
    psi_loo = np.array([float(qteg_sf(t0, loo_alpha[i], loo_beta[i]))
                        for i in range(n)])
    V = n * psi_hat - (n-1) * psi_loo
    return dict(V=V, psi_hat=psi_hat, psi_jack=float(np.mean(V)), n=n)


def survival_jel_ci(V_surv, nominal):
    """
    JEL and AJEL CIs for S(t0) via grid search over (0,1).  [Eq. 41-43]
    """
    cutoff = chi2.ppf(nominal, df=1)
    grid   = np.linspace(1e-6, 1.0 - 1e-6, 1000)
    jvals  = np.array([jel_ratio(V_surv,  g) for g in grid])
    avals  = np.array([ajel_ratio(V_surv, g) for g in grid])

    def _extract(vals):
        inside = grid[vals <= cutoff]
        if len(inside) < 2:
            return (np.nan, np.nan)
        return (float(inside[0]), float(inside[-1]))

    return _extract(jvals), _extract(avals)

# ── Checkpoint helpers ────────────────────────────────────────────────

def save_partial(block_id, counts, rep, n_sim, meta):
    nc = max(counts['n_conv'], 1)
    partial = {**meta, 'status': 'partial', 'rep_done': rep+1,
               'n_sim': n_sim,
               **{k: (100*v/nc if k.startswith('cp_') else v/nc)
                  for k, v in counts.items() if k != 'n_conv'},
               'n_conv': counts['n_conv'],
               '_raw_counts': counts,           # raw integers for resume
               'timestamp': datetime.now().isoformat()}
    fname = os.path.join(RESULTS_DIR, f"block_{block_id:02d}_partial.json")
    with open(fname, 'w') as f:
        json.dump(partial, f, indent=2)


def finalise(block_id, counts, n_sim, meta):
    nc = max(counts['n_conv'], 1)
    result = {**meta, 'status': 'complete', 'n_sim': n_sim,
              'n_conv': counts['n_conv'],
              **{k: (100*v/nc if k.startswith('cp_') else v/nc)
                 for k, v in counts.items() if k != 'n_conv'},
              'timestamp': datetime.now().isoformat()}
    final   = os.path.join(RESULTS_DIR, f"block_{block_id:02d}.json")
    partial = os.path.join(RESULTS_DIR, f"block_{block_id:02d}_partial.json")
    with open(final, 'w') as f:
        json.dump(result, f, indent=2)
    if os.path.exists(partial):
        os.remove(partial)
    return result

# ── Main simulation block ─────────────────────────────────────────────

def run_block(block_id, n_sim=N_SIM_DEFAULT):

    sc_idx  = block_id // 12
    rem     = block_id % 12
    nom_idx = rem // 4
    n_idx   = rem % 4

    alpha_t, beta_t = NULL_SCENARIOS[sc_idx]
    nominal         = NOMINALS[nom_idx]
    n               = SAMPLE_SIZES[n_idx]
    t0              = T0_VALS[(alpha_t, beta_t)]
    psi_true        = float(qteg_sf(t0, alpha_t, beta_t))
    seed            = BASE_SEED + block_id * 10000

    meta = dict(block_id=block_id, scenario=sc_idx+1,
                alpha_t=alpha_t, beta_t=beta_t,
                nominal=nominal, n=n, t0=t0, psi_true=psi_true)

    print(f"\n{'='*60}")
    print(f"Block {block_id:02d} | Sc.{sc_idx+1}(α={alpha_t},β={beta_t}) | "
          f"nominal={nominal:.0%} | n={n}")
    print(f"t0={t0:.4f}  S(t0)={psi_true:.4f}  seed={seed}  n_sim={n_sim}")
    print(f"{'='*60}")

    counts = {k: 0 for k in [
        'n_conv',
        'cp_na_a','al_na_a','cp_jel_a','al_jel_a',
        'cp_ajel_a','al_ajel_a','cp_ejel_a','al_ejel_a',
        'cp_na_b','al_na_b','cp_jel_b','al_jel_b',
        'cp_ajel_b','al_ajel_b','cp_ejel_b','al_ejel_b',
        'cp_na_s','al_na_s',
        'cp_jel_s','al_jel_s','cp_ajel_s','al_ajel_s',
    ]}

    # ── Resume from checkpoint if one exists ─────────────────────────
    resume_from = 0
    partial_path = os.path.join(RESULTS_DIR, f"block_{block_id:02d}_partial.json")
    if os.path.exists(partial_path):
        try:
            with open(partial_path) as f:
                ckpt = json.load(f)
            raw      = ckpt.get('_raw_counts')
            rep_done = ckpt.get('rep_done', 0)
            if raw and rep_done > 0:
                if all(k in raw for k in counts):
                    counts.update({k: int(raw[k]) for k in counts})
                    resume_from = rep_done
                    print(f"  *** Resuming from checkpoint: rep {resume_from}/{n_sim} "
                          f"(conv so far: {counts['n_conv']}) ***")
                else:
                    print(f"  WARNING: Checkpoint missing keys -- starting from rep 0")
            else:
                print(f"  NOTE: Partial file has no _raw_counts -- starting from rep 0")
                print(f"  (Expected for checkpoints saved before v4)")
        except Exception as e:
            print(f"  WARNING: Could not load checkpoint ({e}) -- starting from rep 0")

    t_start = time.time()

    for rep in range(resume_from, n_sim):
        rng = np.random.default_rng(seed + rep)
        y   = qteg_sample(alpha_t, beta_t, n, rng)

        # ── Pseudo-values (single LOO pass) ───────────────────────────
        pv = compute_pseudovalues(y)
        if pv is None:
            continue
        counts['n_conv'] += 1

        ah     = pv['alpha']
        bh     = pv['beta']
        se_a   = pv['se_alpha']
        se_b   = pv['se_beta']
        cov_ab = pv['cov_ab']          # Cov(alpha_hat, beta_hat)
        pva    = pv['pv_alpha']
        pvb    = pv['pv_beta']

        # ── NA intervals for alpha and beta ───────────────────────────
        if np.isfinite(se_a):
            lo, hi = na_ci(ah, se_a, nominal)
            if lo > hi: lo, hi = hi, lo
            if lo <= alpha_t <= hi: counts['cp_na_a'] += 1
            counts['al_na_a'] += hi - lo
        if np.isfinite(se_b):
            lo, hi = na_ci(bh, se_b, nominal)
            if lo > hi: lo, hi = hi, lo
            if lo <= beta_t <= hi: counts['cp_na_b'] += 1
            counts['al_na_b'] += hi - lo

        # ── JEL / AJEL / EJEL for alpha ───────────────────────────────
        for fn, key in [(jel_ci,'jel_a'), (ajel_ci,'ajel_a')]:
            lo, hi = fn(pva, nominal, ah)
            if np.isfinite(lo) and np.isfinite(hi):
                if lo > hi: lo, hi = hi, lo
                if lo <= alpha_t <= hi: counts[f'cp_{key}'] += 1
                counts[f'al_{key}'] += hi - lo

        lo, hi = ejel_ci(pva, nominal, ah)
        if np.isfinite(lo) and np.isfinite(hi):
            if lo > hi: lo, hi = hi, lo
            if lo <= alpha_t <= hi: counts['cp_ejel_a'] += 1
            counts['al_ejel_a'] += hi - lo

        # ── JEL / AJEL / EJEL for beta ────────────────────────────────
        for fn, key in [(jel_ci,'jel_b'), (ajel_ci,'ajel_b')]:
            lo, hi = fn(pvb, nominal, bh)
            if np.isfinite(lo) and np.isfinite(hi):
                if lo > hi: lo, hi = hi, lo
                if lo <= beta_t <= hi: counts[f'cp_{key}'] += 1
                counts[f'al_{key}'] += hi - lo

        lo, hi = ejel_ci(pvb, nominal, bh)
        if np.isfinite(lo) and np.isfinite(hi):
            if lo > hi: lo, hi = hi, lo
            if lo <= beta_t <= hi: counts['cp_ejel_b'] += 1
            counts['al_ejel_b'] += hi - lo

        # ── Survival: NA (full delta method), JEL, AJEL ──────────────
        sv = survival_pseudovalues_from_pv(pv, t0)

        # NA survival CI — now passes cov_ab for full delta method
        if np.isfinite(se_a) and np.isfinite(se_b):
            nlo, nhi = na_survival_ci(ah, bh, se_a, se_b, t0, nominal,
                                      cov_ab=cov_ab)
            if np.isfinite(nlo) and np.isfinite(nhi):
                if nlo > nhi: nlo, nhi = nhi, nlo
                if nlo <= psi_true <= nhi: counts['cp_na_s'] += 1
                counts['al_na_s'] += nhi - nlo

        # JEL and AJEL survival CIs via grid search over (0,1)
        (jlo, jhi), (alo, ahi) = survival_jel_ci(sv['V'], nominal)
        if np.isfinite(jlo) and np.isfinite(jhi):
            if jlo > jhi: jlo, jhi = jhi, jlo
            if jlo <= psi_true <= jhi: counts['cp_jel_s'] += 1
            counts['al_jel_s'] += jhi - jlo
        if np.isfinite(alo) and np.isfinite(ahi):
            if alo > ahi: alo, ahi = ahi, alo
            if alo <= psi_true <= ahi: counts['cp_ajel_s'] += 1
            counts['al_ajel_s'] += ahi - alo

        # ── Progress logging + partial checkpoint ─────────────────────
        if (rep + 1) % CHECKPOINT_EVERY == 0:
            elapsed = time.time() - t_start
            rate    = elapsed / (rep + 1 - resume_from)
            remain  = rate * (n_sim - rep - 1)
            nc      = max(counts['n_conv'], 1)
            print(f"  Rep {rep+1:>5}/{n_sim}  "
                  f"conv={counts['n_conv']:>5}  "
                  f"elapsed={elapsed:>6.0f}s  "
                  f"remain~{remain:>6.0f}s  "
                  f"CP(α)JEL={100*counts['cp_jel_a']/nc:>5.1f}%  "
                  f"CP(S)JEL={100*counts['cp_jel_s']/nc:>5.1f}%  "
                  f"CP(S)NA={100*counts['cp_na_s']/nc:>5.1f}%")
            save_partial(block_id, counts, rep, n_sim, meta)

    result = finalise(block_id, counts, n_sim, meta)

    # Block summary
    print(f"\n--- Block {block_id:02d} complete ---")
    print(f"  Conv: {counts['n_conv']}/{n_sim} "
          f"({100*counts['n_conv']/n_sim:.1f}%)")
    print(f"  CP(α): NA={result['cp_na_a']:.2f}  "
          f"JEL={result['cp_jel_a']:.2f}  "
          f"AJEL={result['cp_ajel_a']:.2f}  "
          f"EJEL={result['cp_ejel_a']:.2f}")
    print(f"  AL(α): NA={result['al_na_a']:.4f}  "
          f"JEL={result['al_jel_a']:.4f}  "
          f"AJEL={result['al_ajel_a']:.4f}  "
          f"EJEL={result['al_ejel_a']:.4f}")
    print(f"  CP(β): NA={result['cp_na_b']:.2f}  "
          f"JEL={result['cp_jel_b']:.2f}  "
          f"AJEL={result['cp_ajel_b']:.2f}  "
          f"EJEL={result['cp_ejel_b']:.2f}")
    print(f"  AL(β): NA={result['al_na_b']:.4f}  "
          f"JEL={result['al_jel_b']:.4f}  "
          f"AJEL={result['al_ajel_b']:.4f}  "
          f"EJEL={result['al_ejel_b']:.4f}")
    print(f"  CP(S): NA={result['cp_na_s']:.2f}  "
          f"JEL={result['cp_jel_s']:.2f}  "
          f"AJEL={result['cp_ajel_s']:.2f}")
    print(f"  AL(S): NA={result['al_na_s']:.4f}  "
          f"JEL={result['al_jel_s']:.4f}  "
          f"AJEL={result['al_ajel_s']:.4f}")
    print(f"  Total time: {time.time()-t_start:.0f}s")
    saved = os.path.join(RESULTS_DIR, f"block_{block_id:02d}.json")
    print(f"  Saved: {saved}")


# ══════════════════════════════════════════════════════════════════════
# REAL DATA APPLICATION
# Datasets: DS1 Bladder Cancer, DS2 Boeing 720, DS3 Melanoma, DS4 Guinea Pig
# ══════════════════════════════════════════════════════════════════════

import re as _re

def _parse(raw):
    return np.array([float(x) for x in _re.findall(r'\d+\.?\d*', raw)])

DATASETS = {
    'DS1: Bladder Cancer (n=128)': _parse(
        "0.08,2.09,3.48,4.87,6.94,8.66,13.11,23.63,0.20,2.23,3.52,4.98,"
        "6.97,9.02,13.29,0.40,2.26,3.57,5.06,7.09,9.22,13.80,25.74,0.50,"
        "2.46,3.64,5.09,7.26,9.47,14.24,25.82,0.51,2.54,3.70,5.17,7.28,"
        "9.74,14.76,26.31,0.81,2.62,3.82,5.32,7.32,10.06,14.77,32.15,2.64,"
        "11.79,18.10,1.46,4.40,5.85,8.26,11.98,19.13,1.76,3.25,4.50,6.25,"
        "8.37,12.02,2.02,3.31,4.51,6.54,8.53,12.03,20.28,2.02,3.36,6.76,"
        "12.07,21.73,2.0,3.36,6.93,8.65,12.63,22.69,3.88,5.32,7.39,10.34,"
        "14.83,34.26,0.90,2.69,4.18,5.34,7.59,10.66,15.96,36.66,1.05,2.69,"
        "4.23,5.41,7.62,10.75,16.62,43.01,1.19,2.75,4.26,5.41,7.63,17.12,"
        "46.12,1.26,2.83,4.33,5.49,7.66,11.25,17.14,79.05,1.35,2.87,5.62,"
        "7.87,11.64,17.36,1.40,3.02,4.34,5.71,7.93"),

    'DS2: Boeing 720 (n=213)': _parse(
        "194,413,90,74,55,23,97,50,359,50,130,487,102,15,14,10,57,320,261,"
        "51,44,9,254,493,18,209,41,58,60,48,56,87,11,102,12,5,100,14,29,"
        "37,186,29,104,7,4,72,270,283,7,57,33,100,61,502,220,120,141,22,"
        "603,35,98,54,181,65,49,12,239,14,18,39,3,12,5,32,9,14,70,47,62,"
        "142,3,104,85,67,169,24,21,246,47,68,15,2,91,59,447,56,29,176,225,"
        "77,197,438,43,134,184,20,386,182,71,80,188,230,152,36,79,59,33,"
        "246,1,79,3,27,201,84,27,21,16,88,130,14,118,44,15,42,106,46,230,"
        "59,153,104,20,206,5,66,34,29,26,35,5,82,5,61,31,118,326,12,54,"
        "36,34,18,25,120,31,22,18,156,11,216,139,67,310,3,46,210,57,76,"
        "14,111,97,62,26,71,39,30,7,44,11,63,23,22,23,14,18,13,34,62,11,"
        "191,14,16,18,130,90,163,208,1,24,70,16,101,52,208,95"),

    'DS3: Malignant Melanoma (n=205)': _parse(
        "6.76,0.65,1.34,2.90,12.08,4.84,5.16,3.22,12.88,7.41,4.19,0.16,3.87,"
        "4.84,2.42,12.56,5.80,7.06,5.48,7.73,13.85,2.34,4.19,4.04,4.84,0.32,"
        "8.54,2.58,3.56,3.54,0.97,4.83,1.62,6.44,14.66,2.58,3.87,3.54,1.34,"
        "2.24,3.87,3.54,17.42,1.29,3.22,1.29,4.51,8.38,1.94,0.16,2.58,1.29,"
        "0.16,1.62,1.29,2.10,0.32,0.81,1.13,5.16,1.62,1.37,0.24,0.81,1.29,"
        "1.29,0.97,1.13,5.80,1.29,0.48,1.62,2.26,0.58,0.97,2.58,0.81,3.54,"
        "0.97,1.78,1.94,1.29,3.22,1.53,1.29,1.62,1.62,0.32,4.84,1.29,0.97,"
        "3.06,3.54,1.62,2.58,1.94,0.81,7.73,0.97,12.88,2.58,4.09,0.64,0.97,"
        "3.22,1.62,3.87,0.32,0.32,3.22,2.26,3.06,2.58,0.65,1.13,0.81,0.97,"
        "1.76,1.94,0.65,0.97,5.64,9.66,0.10,5.48,2.26,4.83,0.97,0.97,5.16,"
        "0.81,2.90,3.87,1.94,0.16,0.64,2.26,1.45,4.82,1.29,7.89,0.81,3.54,"
        "1.29,0.64,3.22,1.45,0.48,1.94,0.16,0.16,1.29,1.94,3.54,0.81,0.65,"
        "7.09,0.16,1.62,1.62,1.29,6.12,0.48,0.64,3.22,1.94,2.58,2.58,0.81,"
        "0.81,3.22,0.32,3.22,2.74,4.84,1.62,0.65,1.45,0.65,1.29,1.62,3.54,"
        "3.22,0.65,1.03,7.09,1.29,0.65,1.78,12.24,8.06,0.81,2.10,3.87,0.65,"
        "1.94,0.65,2.10,1.94,1.13,7.06,6.12,0.48,2.26,2.90"),

    'DS4: Guinea Pig Survival (n=72)': _parse(
        "0.1,0.33,0.44,0.56,0.59,0.72,0.74,0.77,0.92,0.93,0.96,1.0,"
        "1.0,1.02,1.05,1.07,1.07,1.08,1.08,1.09,1.12,1.13,1.15,1.16,"
        "1.2,1.21,1.22,1.22,1.24,1.3,1.34,1.36,1.39,1.44,1.46,1.53,"
        "1.59,1.6,1.63,1.68,1.71,1.72,1.76,1.83,1.95,1.96,1.97,2.02,"
        "2.13,2.15,2.16,2.22,2.3,2.31,2.4,2.45,2.51,2.53,2.54,2.54,"
        "2.78,2.93,3.27,3.42,3.47,3.61,4.02,4.32,4.58,5.55,6.0,9.4"),
}

DATASET_T0 = {
    'DS1: Bladder Cancer (n=128)':      4.0,
    'DS2: Boeing 720 (n=213)':          50.0,
    'DS3: Malignant Melanoma (n=205)':  1.5,
    'DS4: Guinea Pig Survival (n=72)':  1.5,
}


def run_real_data(nominal=0.95):
    """
    For each dataset: compute MLE Wald CIs, JEL, AJEL, EJEL CIs for alpha
    and beta, and NA/JEL/AJEL CIs for the survival function at t0.
    NA survival CI uses full covariance delta method (cov_ab).
    """
    print("\n\n" + "="*90)
    print("TABLE 5 -- REAL DATA: MLE ESTIMATES AND JEL-BASED CONFIDENCE INTERVALS")
    print(f"  Nominal level: {int(nominal*100)}%")
    print("="*90)

    all_results = {}

    for ds_name, y in DATASETS.items():
        t0 = DATASET_T0[ds_name]
        print(f"\n{'─'*90}")
        print(f"  {ds_name}   (t0 = {t0})")
        print(f"{'─'*90}")

        fit = qteg_mle(y)
        if fit is None:
            print("  MLE failed. Skipping.")
            continue

        ah, bh   = fit['alpha'], fit['beta']
        se_a     = fit['se_alpha']
        se_b     = fit['se_beta']
        cov_ab   = fit['cov_ab']

        print(f"  MLE:  alpha_hat = {ah:.4f} (SE = {se_a:.4f}),  "
              f"beta_hat = {bh:.4f} (SE = {se_b:.4f}),  "
              f"Cov = {cov_ab:.6f}")
        print(f"  logL = {fit['logL']:.4f}")

        print(f"  Computing {len(y)} leave-one-out MLEs ...", end=' ', flush=True)
        t_start = time.time()
        pv = compute_pseudovalues(y)
        elapsed = time.time() - t_start
        print(f"done ({elapsed:.1f}s)")

        if pv is None:
            print("  Pseudo-value computation failed. Skipping.")
            continue

        pva = pv['pv_alpha']
        pvb = pv['pv_beta']

        # CIs for alpha
        na_a   = na_ci(ah, se_a, nominal)
        jel_a  = jel_ci(pva, nominal, ah)
        ajel_a = ajel_ci(pva, nominal, ah)
        ejel_a = ejel_ci(pva, nominal, ah)

        # CIs for beta
        na_b   = na_ci(bh, se_b, nominal)
        jel_b  = jel_ci(pvb, nominal, bh)
        ajel_b = ajel_ci(pvb, nominal, bh)
        ejel_b = ejel_ci(pvb, nominal, bh)

        # Survival CIs — NA uses full cov_ab
        sv_pv  = survival_pseudovalues_from_pv(pv, t0)
        Vs     = sv_pv['V']
        cutoff = chi2.ppf(nominal, df=1)
        grid_s = np.linspace(1e-6, 1.0 - 1e-6, 1000)
        jvals  = np.array([jel_ratio(Vs, g)  for g in grid_s])
        avals  = np.array([ajel_ratio(Vs, g) for g in grid_s])

        def _extr(vals):
            inside = grid_s[vals <= cutoff]
            return (float(inside[0]), float(inside[-1])) \
                   if len(inside) >= 2 else (np.nan, np.nan)

        na_s = na_survival_ci(ah, bh, se_a, se_b, t0, nominal,
                               cov_ab=cov_ab)

        sv = dict(psi_hat=sv_pv['psi_hat'],
                  psi_jack=sv_pv['psi_jack'],
                  na=na_s, jel=_extr(jvals), ajel=_extr(avals))

        def fmt_ci(ci):
            if np.isnan(ci[0]):
                return "    [  ---  ,   ---  ]"
            return f"    [{ci[0]:7.4f}, {ci[1]:7.4f}]  (width={ci[1]-ci[0]:.4f})"

        print(f"\n  Confidence intervals for alpha ({int(nominal*100)}%):")
        print(f"    NA:    {fmt_ci(na_a)}")
        print(f"    JEL:   {fmt_ci(jel_a)}")
        print(f"    AJEL:  {fmt_ci(ajel_a)}")
        print(f"    EJEL:  {fmt_ci(ejel_a)}")

        print(f"\n  Confidence intervals for beta ({int(nominal*100)}%):")
        print(f"    NA:    {fmt_ci(na_b)}")
        print(f"    JEL:   {fmt_ci(jel_b)}")
        print(f"    AJEL:  {fmt_ci(ajel_b)}")
        print(f"    EJEL:  {fmt_ci(ejel_b)}")

        psi_hat = sv['psi_hat']
        print(f"\n  Survival function S({t0}) = {psi_hat:.4f}")
        print(f"  NA  CI for S({t0}):   {fmt_ci(sv['na'])}")
        print(f"  JEL CI for S({t0}):   {fmt_ci(sv['jel'])}")
        print(f"  AJEL CI for S({t0}):  {fmt_ci(sv['ajel'])}")

        all_results[ds_name] = dict(
            fit=fit, pv=pv,
            ci_alpha=dict(na=na_a, jel=jel_a, ajel=ajel_a, ejel=ejel_a),
            ci_beta=dict(na=na_b, jel=jel_b, ajel=ajel_b, ejel=ejel_b),
            survival=sv, t0=t0
        )

    return all_results


def run_and_save_real_data(nominal=0.95):
    """Run real data application and save results to JSON."""
    import json as _json
    import numpy as _np

    results = run_real_data(nominal=nominal)
    if not results:
        return results

    def _ser(obj):
        if isinstance(obj, _np.ndarray): return obj.tolist()
        if isinstance(obj, (_np.floating, float)): return float(obj)
        if isinstance(obj, (_np.integer, int)): return int(obj)
        if isinstance(obj, dict): return {k: _ser(v) for k,v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_ser(i) for i in obj]
        return obj

    out = os.path.join(RESULTS_DIR, 'QTEG_JEL_realdata_results.json')
    with open(out, 'w') as f:
        _json.dump(_ser(results), f, indent=2)
    print(f"Real data results saved: {out}")
    return results

# ── Merge results ─────────────────────────────────────────────────────

def merge_results():
    results = []
    missing = []
    for i in range(36):
        fname = os.path.join(RESULTS_DIR, f"block_{i:02d}.json")
        if os.path.exists(fname):
            with open(fname) as f:
                r = json.load(f)
            if r.get('status') == 'complete':
                results.append(r)
            else:
                print(f"  WARNING: block {i:02d} status={r.get('status')}")
                missing.append(i)
        else:
            pname = os.path.join(RESULTS_DIR, f"block_{i:02d}_partial.json")
            if os.path.exists(pname):
                print(f"  Block {i:02d}: partial only")
            else:
                missing.append(i)

    if missing:
        print(f"\nMissing/incomplete blocks: {missing}")
    else:
        print(f"\nAll 36 blocks complete.")

    json_out = os.path.join(RESULTS_DIR, 'QTEG_JEL_full_results.json')
    with open(json_out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Merged {len(results)} blocks → {json_out}")

    if results:
        csv_out = os.path.join(RESULTS_DIR, 'QTEG_JEL_full_results.csv')
        keys = list(results[0].keys())
        with open(csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV summary → {csv_out}")

    print(f"\n{'Sc':<4} {'Nom':>5} {'n':>4}  "
          f"{'CP_α NA':>8} {'CP_α JEL':>9} {'CP_α AJEL':>10}  "
          f"{'AL_α JEL':>9}  {'CP_S NA':>8} {'CP_S JEL':>9}")
    print("-"*80)
    for r in sorted(results,
                    key=lambda x: (x['scenario'], x['nominal'], x['n'])):
        print(f"Sc.{r['scenario']} {r['nominal']:>5.0%} {r['n']:>4}  "
              f"{r['cp_na_a']:>8.2f} {r['cp_jel_a']:>9.2f} "
              f"{r['cp_ajel_a']:>10.2f}  "
              f"{r['al_jel_a']:>9.4f}  "
              f"{r.get('cp_na_s', float('nan')):>8.2f} "
              f"{r['cp_jel_s']:>9.2f}")

# ── Entry point ───────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='QTEG JEL simulation — Arctic HPC (v4)')
    parser.add_argument('--block', type=int, default=None,
                        help='Block index 0-35')
    parser.add_argument('--merge', action='store_true',
                        help='Merge all completed block results')
    parser.add_argument('--n_sim', type=int, default=N_SIM_DEFAULT,
                        help='Replications per block (default 5000)')
    parser.add_argument('--realdata', action='store_true',
                        help='Run real data application and save JSON')
    parser.add_argument('--nominal', type=float, default=0.95,
                        help='Nominal level for real data (default 0.95)')
    parser.add_argument('--test', action='store_true',
                        help='Quick test: block 0, 20 reps')
    args = parser.parse_args()

    if args.realdata:
        run_and_save_real_data(nominal=args.nominal)
    elif args.test:
        print("TEST MODE: block 0, 20 replications")
        run_block(0, n_sim=20)
    elif args.merge:
        merge_results()
    elif args.block is not None:
        run_block(args.block, n_sim=args.n_sim)
    else:
        print("Usage:")
        print("  Test:         python QTEG_JEL_Arctic.py --test")
        print("  Single block: python QTEG_JEL_Arctic.py --block 0")
        print("  Custom reps:  python QTEG_JEL_Arctic.py --block 0 --n_sim 100")
        print("  Merge:        python QTEG_JEL_Arctic.py --merge")
        print("  Via SLURM:    sbatch qteg_jel_array.sh")
        print("  Real data:    python QTEG_JEL_Arctic.py --realdata")
