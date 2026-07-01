"""
Neutrality test for donor→recipient transmission experiments.

Model:
- Donor composition defines a baseline frequency p_j for each feature j.
- Each recipient is modeled as a Dirichlet–multinomial (DM) sample with
  concentration Nb_m (the "bottleneck size" / effective population size)
  and center q_j.
- For each feature j, we test neutrality:
      H0: q_j = p_j   vs   H1: q_j free
  using a likelihood-ratio test (LRT) with chi-square(1) calibration.

This module is extracted/generalized from Neutrality_Test_Dox_UPDATED.ipynb.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests


_MIN_PVAL = float(np.nextafter(0.0, 1.0))  # smallest positive float64


def _log_beta(a: float, b: float) -> float:
    return float(gammaln(a) + gammaln(b) - gammaln(a + b))


def _log_bb_pmf(x: int, n: int, a: float, b: float) -> float:
    """
    Beta-binomial log pmf:
        X | n, a, b ~ BetaBinomial(n, a, b)
    """
    return float(
        (gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1))
        + _log_beta(x + a, n - x + b)
        - _log_beta(a, b)
    )


def _loglik_feature(
    q: float,
    x_vec: np.ndarray,
    n_vec: np.ndarray,
    Nb_vec: np.ndarray,
    *,
    alpha0: float,
    q_bounds: Tuple[float, float],
) -> float:
    """
    Marginal DM log-likelihood contribution for a single feature across recipients,
    using the fact that each feature marginal is beta-binomial under DM.

    Parameters
    ----------
    q : float
        Center frequency in recipients (clipped to q_bounds).
    x_vec : (M,) int
        Counts for the feature in each recipient.
    n_vec : (M,) int
        Total counts in each recipient.
    Nb_vec : (M,) float
        Bottleneck/concentration per recipient.
    alpha0 : float
        Small additive term to avoid zeros in concentration parameters.
    q_bounds : (lo, hi)
        Bounds for q to avoid log(0) issues.
    """
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


def _fit_q_mle(
    x_vec: np.ndarray,
    n_vec: np.ndarray,
    Nb_vec: np.ndarray,
    *,
    alpha0: float,
    q_bounds: Tuple[float, float],
) -> Tuple[float, float]:
    """
    MLE for q for one feature. Returns (q_hat, loglik_at_hat).
    """
    lo, hi = q_bounds

    def nll(q: float) -> float:
        return -_loglik_feature(q, x_vec, n_vec, Nb_vec, alpha0=alpha0, q_bounds=q_bounds)

    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded")
    q_hat = float(res.x)
    ll_hat = float(-res.fun)
    return q_hat, ll_hat


def _as_1d_float_array(x: Union[float, Sequence[float], np.ndarray], length: int) -> np.ndarray:
    if np.isscalar(x):
        return np.full(length, float(x), dtype=float)
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"bottleneck has length {arr.shape[0]}, expected {length}")
    return arr


def neutrality_lrt(
    donor_counts: Union[pd.Series, Sequence[float], np.ndarray],
    recipient_counts: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[float]]],
    bottleneck: Union[float, Sequence[float], np.ndarray],
    *,
    feature_ids: Optional[Sequence[object]] = None,
    pseudocount: float = 1e-6,
    alpha0: float = 1e-12,
    min_p: float = 1e-8,
    q_bounds: Tuple[float, float] = (1e-12, 1 - 1e-12),
    fdr_alpha: float = 0.05,
    fdr_method: str = "fdr_bh",
) -> pd.DataFrame:
    """
    Run the neutrality test for all features.

    Parameters
    ----------
    donor_counts
        Raw counts at the donor timepoint (e.g., inoculum), shape (K,).
        Used only to define p_donor via normalization with a pseudocount.
    recipient_counts
        Raw counts at the recipient timepoint(s), shape (M, K) where
        rows correspond to recipients/replicates and columns to features.
        If a DataFrame is provided, its columns are used as feature IDs
        unless feature_ids is supplied.
    bottleneck
        Effective bottleneck size Nb. Either a scalar (applied to all M)
        or length-M array-like (one Nb per recipient/replicate).
    feature_ids
        Optional length-K feature labels. If None, tries to use:
          - donor_counts.index if donor_counts is a pandas Series
          - recipient_counts.columns if recipient_counts is a DataFrame
          - otherwise uses 0..K-1
    pseudocount
        Added to donor counts before normalization to avoid zeros.
    alpha0
        Small additive term to avoid exactly zero DM concentration parameters.
    min_p
        Features with p_donor < min_p or > 1-min_p are not tested (pval=NaN).
    q_bounds
        Bounds for q during optimization and likelihood evaluation.
    fdr_alpha
        Target FDR for rejection flag.
    fdr_method
        Method passed to statsmodels.stats.multitest.multipletests.

    Returns
    -------
    pandas.DataFrame with columns:
        feature, p_donor, q_hat, LR, pval, qval_FDR, reject_FDR, direction
    """
    # Recipient matrix
    if isinstance(recipient_counts, pd.DataFrame):
        X = recipient_counts.to_numpy(dtype=np.int64)
        cols_from_recip = list(recipient_counts.columns)
    else:
        X = np.asarray(recipient_counts, dtype=np.int64)
        cols_from_recip = None

    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError("recipient_counts must be 1D or 2D with shape (M, K)")
    M, K = X.shape

    # Donor vector
    if isinstance(donor_counts, pd.Series):
        donor_arr = donor_counts.to_numpy(dtype=float).reshape(-1)
        ids_from_donor = list(donor_counts.index)
    else:
        donor_arr = np.asarray(donor_counts, dtype=float).reshape(-1)
        ids_from_donor = None

    if donor_arr.shape[0] != K:
        raise ValueError(f"donor_counts length {donor_arr.shape[0]} does not match K={K}")

    # Feature IDs
    if feature_ids is None:
        if cols_from_recip is not None:
            feature_ids = cols_from_recip
        elif ids_from_donor is not None:
            feature_ids = ids_from_donor
        else:
            feature_ids = list(range(K))
    if len(feature_ids) != K:
        raise ValueError(f"feature_ids length {len(feature_ids)} does not match K={K}")

    # Nb vector and per-recipient totals
    Nb_vec = _as_1d_float_array(bottleneck, M)
    n_tot = X.sum(axis=1).astype(np.int64)

    # Donor frequencies
    donor_adj = donor_arr + float(pseudocount)
    if np.any(donor_adj < 0):
        raise ValueError("donor_counts must be nonnegative")
    p_donor = donor_adj / donor_adj.sum()

    # LRT per feature
    q_hat = np.full(K, np.nan, dtype=float)
    LR = np.full(K, np.nan, dtype=float)
    pval = np.full(K, np.nan, dtype=float)

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

        p = float(chi2.sf(LRj, df=1))  # stable tail probability
        if (not np.isfinite(p)) or (p <= 0.0):
            p = _MIN_PVAL
        pval[j] = p

    # Multiple testing correction (FDR)
    tested = np.isfinite(pval)
    qval = np.full(K, np.nan, dtype=float)
    reject = np.zeros(K, dtype=bool)

    if tested.any():
        r, qv, _, _ = multipletests(pval[tested], alpha=float(fdr_alpha), method=str(fdr_method))
        qval[tested] = np.maximum(qv, _MIN_PVAL)
        pval[tested] = np.maximum(pval[tested], _MIN_PVAL)
        reject[tested] = r

    res = pd.DataFrame(
        {
            "feature": list(feature_ids),
            "p_donor": p_donor,
            "q_hat_recipient_center": q_hat,
            "LR": LR,
            "pval": pval,
            "qval_FDR": qval,
            "reject_FDR": reject,
        }
    )

    direction = pd.Series([None]*len(res), dtype="object")
    mask = np.isfinite(res["q_hat_recipient_center"].to_numpy(dtype=float))
    direction.loc[mask] = np.where(
        res.loc[mask, "q_hat_recipient_center"] > res.loc[mask, "p_donor"],
        "up_in_recipient",
        "down_in_recipient",
    )
    res["direction"] = direction
    return res.sort_values(["pval", "LR"], ascending=[True, False]).reset_index(drop=True)
