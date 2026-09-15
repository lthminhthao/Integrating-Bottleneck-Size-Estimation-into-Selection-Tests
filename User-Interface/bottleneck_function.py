"""
Bottleneck (effective population size Nb) estimation via Dirichlet-Multinomial MLE.
Extracted from bottleneck_function.ipynb.
"""
import numpy as np
from scipy.special import gammaln


def donor_frequencies(counts_t0, pseudocount=1e-6):
    """Convert raw donor counts at time 0 into a probability vector p (with pseudocount)."""
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


def dm_loglik(counts_t1, Nb, p, alpha0=1e-12):
    """Dirichlet-multinomial log-likelihood for counts x given Nb and base probabilities p."""
    x = np.asarray(counts_t1, dtype=np.int64)
    if x.ndim != 1:
        raise ValueError("counts_t1 must be a 1D vector.")
    if np.any(x < 0):
        raise ValueError("counts_t1 must be nonnegative.")
    n = int(x.sum())
    alpha = Nb * np.asarray(p, dtype=float) + float(alpha0)
    A = float(alpha.sum())
    return float((gammaln(n + 1) - np.sum(gammaln(x + 1)))
                 + (gammaln(A) - gammaln(A + n))
                 + np.sum(gammaln(alpha + x) - gammaln(alpha)))


def bottleneck_mle_from_p(counts_t1, p, nb_max=5_000_000, n_grid=220, refine=220, alpha0=1e-12):
    """MLE of Nb given recipient counts and donor probability vector p (two-stage grid search)."""
    x = np.asarray(counts_t1, dtype=np.int64)
    if x.sum() == 0:
        return np.nan
    grid = np.unique(np.round(np.logspace(0, np.log10(nb_max), n_grid)).astype(int))
    ll = np.array([dm_loglik(x, Nb, p, alpha0=alpha0) for Nb in grid])
    Nb0 = int(grid[np.argmax(ll)])
    lo = max(1, Nb0 // 2)
    hi = min(nb_max, Nb0 * 2)
    grid2 = np.unique(np.linspace(lo, hi, refine).astype(int))
    ll2 = np.array([dm_loglik(x, Nb, p, alpha0=alpha0) for Nb in grid2])
    return int(grid2[np.argmax(ll2)])


def bottleneck_from_two_timepoints(counts_t0, counts_t1,
                                   donor_pseudocount=1e-6, alpha0=1e-12,
                                   nb_max=5_000_000, n_grid=220, refine=220,
                                   return_p=False):
    """Estimate Nb from raw donor counts (t0) and one recipient's raw counts (t1)."""
    p = donor_frequencies(counts_t0, pseudocount=donor_pseudocount)
    nb_hat = bottleneck_mle_from_p(counts_t1, p, nb_max=nb_max, n_grid=n_grid, refine=refine, alpha0=alpha0)
    return (nb_hat, p) if return_p else nb_hat
