#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.special import gammaln


# In[2]:


def donor_frequencies(counts_t0, pseudocount=1e-6):
    """
    Convert raw donor counts at time 0 into a probability vector p (with pseudocount).
    """
    c0 = np.asarray(counts_t0, dtype=float)
    if c0.ndim != 1:
        raise ValueError("counts_t0 must be a 1D vector.")
    if not np.isfinite(c0).all() or np.any(c0 < 0):
        raise ValueError("counts_t0 must be finite and nonnegative.")

    p = c0 + float(pseudocount)
    s = float(p.sum())
    if s <= 0:
        raise ValueError("sum(counts_t0 + pseudocount) must be > 0.")
    return p / s


# ###  Dirichlet–multinomial log-likelihood for counts x given Nb and base probabilities p.
# This mirrors the notebook formula:\
# log(n!) - sum log(x_i!) + log Gamma(A) - log Gamma(A+n) + sum_i [log Gamma(alpha_i + x_i) - log Gamma(alpha_i)]
# where alpha_i = Nb * p_i + alpha0, A = sum_i alpha_i, n = sum_i x_i.

# In[3]:


def dm_loglik(counts_t1, Nb, p, alpha0=1e-12):
    x = np.asarray(counts_t1, dtype=np.int64)
    if x.ndim != 1:
        raise ValueError("counts_t1 must be a 1D vector.")
    if np.any(x < 0):
        raise ValueError("counts_t1 must be nonnegative.")

    n = int(x.sum())
    alpha = Nb * np.asarray(p, dtype=float) + float(alpha0)
    A = float(alpha.sum())

    return float((gammaln(n + 1) - np.sum(gammaln(x + 1))) + (gammaln(A) - gammaln(A + n)) + np.sum(gammaln(alpha + x) - gammaln(alpha)))


def bottleneck_mle_from_p(counts_t1, p, nb_max=5_000_000, n_grid=220, refine=220, alpha0=1e-12):
    """
    MLE of Nb given recipient counts at time 1 and donor probability vector p,
    using the same two-stage grid search as the notebook.
    """
    x = np.asarray(counts_t1, dtype=np.int64)
    if x.sum() == 0:
        return np.nan  # no data to identify Nb

    # Coarse grid on a log scale
    grid = np.unique(np.round(np.logspace(0, np.log10(nb_max), n_grid)).astype(int))
    ll = np.array([dm_loglik(x, Nb, p, alpha0=alpha0) for Nb in grid])
    Nb0 = int(grid[np.argmax(ll)])

    # Refine locally around Nb0 (±2x)
    lo = max(1, Nb0 // 2)
    hi = min(nb_max, Nb0 * 2)
    grid2 = np.unique(np.linspace(lo, hi, refine).astype(int))
    ll2 = np.array([dm_loglik(x, Nb, p, alpha0=alpha0) for Nb in grid2])

    return int(grid2[np.argmax(ll2)])


# In[4]:


def bottleneck_from_two_timepoints(counts_t0, counts_t1,
                                   donor_pseudocount=1e-6, alpha0=1e-12,
                                   nb_max=5_000_000, n_grid=220, refine=220,
                                   return_p=False):
    """ - counts_t0: raw counts at time 0 (donor / preinfection) for each sgRNA
        - counts_t1: raw counts at time 1 (recipient) for the same sgRNAs (same ordering)
    Returns Nb_hat (int) or np.nan if counts_t1 sums to 0. """
    p = donor_frequencies(counts_t0, pseudocount=donor_pseudocount)
    nb_hat = bottleneck_mle_from_p(counts_t1, p, nb_max=nb_max, n_grid=n_grid, refine=refine, alpha0=alpha0)
    return (nb_hat, p) if return_p else nb_hat

