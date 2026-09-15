"""
Neutrality test for donor->recipient transmission experiments.
"""
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

_MIN_PVAL = float(np.nextafter(0.0, 1.0))

def _log_beta(a, b):
    return float(gammaln(a) + gammaln(b) - gammaln(a + b))

def _log_bb_pmf(x, n, a, b):
    return float(
        (gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1))
        + _log_beta(x + a, n - x + b)
        - _log_beta(a, b)
    )

def _loglik_feature(q, x_vec, n_vec, Nb_vec, *, alpha0, q_bounds):
    lo, hi = q_bounds
    q = float(np.clip(q, lo, hi))
    ll = 0.0
    for xm, nm, Nbm in zip(x_vec, n_vec, Nb_vec):
        if nm <= 0:
            continue
        a = float(Nbm * q + alpha0)
        b = float(Nbm * (1.0 - q) + alpha0)
        ll += _log_bb_pmf(int(xm), int(nm), a, b)
    return float(ll)

def _fit_q_mle(x_vec, n_vec, Nb_vec, *, alpha0, q_bounds):
    lo, hi = q_bounds
    def nll(q):
        return -_loglik_feature(q, x_vec, n_vec, Nb_vec, alpha0=alpha0, q_bounds=q_bounds)
    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded")
    return float(res.x), float(-res.fun)

def _as_1d_float_array(x, length):
    if np.isscalar(x):
        return np.full(length, float(x), dtype=float)
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"bottleneck has length {arr.shape[0]}, expected {length}")
    return arr

def neutrality_lrt(
    donor_counts, recipient_counts, bottleneck, *,
    feature_ids=None, pseudocount=1e-6, alpha0=1e-12,
    min_p=1e-8, q_bounds=(1e-12, 1 - 1e-12),
    fdr_alpha=0.05, fdr_method="fdr_bh",
):
    if isinstance(recipient_counts, pd.DataFrame):
        X = recipient_counts.to_numpy(dtype=np.int64)
        cols_from_recip = list(recipient_counts.columns)
    else:
        X = np.asarray(recipient_counts, dtype=np.int64)
        cols_from_recip = None
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError("recipient_counts must be 1D or 2D")
    M, K = X.shape

    if isinstance(donor_counts, pd.Series):
        donor_arr = donor_counts.to_numpy(dtype=float).reshape(-1)
        ids_from_donor = list(donor_counts.index)
    else:
        donor_arr = np.asarray(donor_counts, dtype=float).reshape(-1)
        ids_from_donor = None
    if donor_arr.shape[0] != K:
        raise ValueError(f"donor_counts length does not match K={K}")

    if feature_ids is None:
        if cols_from_recip is not None:
            feature_ids = cols_from_recip
        elif ids_from_donor is not None:
            feature_ids = ids_from_donor
        else:
            feature_ids = list(range(K))

    Nb_vec = _as_1d_float_array(bottleneck, M)
    n_tot  = X.sum(axis=1).astype(np.int64)
    donor_adj = donor_arr + float(pseudocount)
    p_donor   = donor_adj / donor_adj.sum()

    q_hat = np.full(K, np.nan)
    LR    = np.full(K, np.nan)
    pval  = np.full(K, np.nan)

    for j in range(K):
        p0 = float(p_donor[j])
        if (not np.isfinite(p0)) or (p0 < min_p) or (p0 > 1.0 - min_p):
            continue
        x_vec = X[:, j].astype(np.int64, copy=False)
        ll0 = _loglik_feature(p0, x_vec, n_tot, Nb_vec, alpha0=alpha0, q_bounds=q_bounds)
        qh, ll1 = _fit_q_mle(x_vec, n_tot, Nb_vec, alpha0=alpha0, q_bounds=q_bounds)
        q_hat[j] = qh
        LRj = max(0.0, 2.0 * (ll1 - ll0))
        LR[j] = LRj
        p = float(chi2.sf(LRj, df=1))
        if (not np.isfinite(p)) or p <= 0.0:
            p = _MIN_PVAL
        pval[j] = p

    tested = np.isfinite(pval)
    qval   = np.full(K, np.nan)
    reject = np.zeros(K, dtype=bool)
    if tested.any():
        r, qv, _, _ = multipletests(pval[tested], alpha=float(fdr_alpha), method=str(fdr_method))
        qval[tested]  = np.maximum(qv, _MIN_PVAL)
        pval[tested]  = np.maximum(pval[tested], _MIN_PVAL)
        reject[tested] = r

    res = pd.DataFrame({
        "feature": list(feature_ids),
        "p_donor": p_donor,
        "q_hat_recipient_center": q_hat,
        "LR": LR,
        "pval": pval,
        "qval_FDR": qval,
        "reject_FDR": reject,
    })
    direction = pd.Series([None]*len(res), dtype="object")
    mask = np.isfinite(res["q_hat_recipient_center"].to_numpy(dtype=float))
    direction.loc[mask] = np.where(
        res.loc[mask, "q_hat_recipient_center"] > res.loc[mask, "p_donor"],
        "up_in_recipient", "down_in_recipient",
    )
    res["direction"] = direction
    return res.sort_values(["pval","LR"], ascending=[True,False]).reset_index(drop=True)
