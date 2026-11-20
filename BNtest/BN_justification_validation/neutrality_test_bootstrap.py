"""Bottleneck bootstrap with a direction-agreement likelihood-ratio statistic.

Use neutrality_bootstrap (or legacy neutrality_lrt_bootstrap).
T = LR when fitted q and equal-host observed frequency are on the same
side of donor p; otherwise T=0. Each bootstrap repeats fitting and gating.
Discordance implies no rejection by this rule, not evidence of neutrality.
This proposed procedure requires calibration studies; it does not guarantee
FDR control or remove compositional effects. Donor uncertainty is omitted.
Dependencies: NumPy, pandas, SciPy >=1.11, C++17 compiler; statsmodels for
corrections other than BH/BY. Likelihood fitting adds runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.special import logsumexp
from scipy.stats import binom, betabinom
from scipy.stats import chi2
def multipletests(pvals, alpha=0.05, method='fdr_bh'):
    """Use SciPy for BH/BY; statsmodels is optional for other corrections."""
    if method in ('fdr_bh', 'fdr_by'):
        from scipy.stats import false_discovery_control
        adjusted = false_discovery_control(pvals, method=method[4:])
        return adjusted <= alpha, adjusted, None, None
    from statsmodels.stats.multitest import multipletests as sm_multipletests
    return sm_multipletests(pvals, alpha=alpha, method=method)




"""Adaptive evaluation of the founder-mixture marginal h_B.

Requires NumPy and SciPy. No compilation required.
rtol controls omitted-sum error, NOT floating-point special-function error.
The latter can exceed rtol for large parameters. No founder-count rounding.

For 1 <= c <= B-1, let w(c) = Binom(c; B,D)*BB(x;n,c,B-c).
Its adjacent ratio is
 (B-c)/(c+1) * D/(1-D) * (1+x/c) / (1+(n-x)/(B-c-1)).
Every c-dependent factor is nonincreasing. Thus w is log-concave.
At either boundary, the outward ratio bounds all subsequent outward ratios,
so an infinite geometric series bounds the omitted finite tail.
Endpoints c=0,B are included exactly. Summation is never renormalized.
"""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom, betabinom


def log_h_B(x, n, D, B, *, rtol=1e-8, return_info=False):
    """Return log h_B; optionally return (value, diagnostics).

    In exact arithmetic, omitted mass / retained mass <= rtol.
    Therefore the log-probability error is <= log1p(rtol).
    Diagnostics report this truncation bound, not rounding error.
    """
    if np.isposinf(B):
        value = float(binom.logpmf(x, n, D))
        if return_info:
            return value, dict(terms_evaluated=0, relative_truncation_bound=0.0, log_error_bound=0.0)
        return value
    for name, value, minimum in [('B', B, 1), ('n', n, 0)]:
        if not np.isfinite(value) or value < minimum or value != np.floor(value):
            raise ValueError(f'{name} must be an integer >= {minimum}.')
    if not np.isfinite(D) or not 0 <= D <= 1:
        raise ValueError('D must lie in [0, 1].')
    if not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError('rtol must lie in (0, 1).')
    B, n, D = int(B), int(n), float(D)

    def result(value, count=0, log_bound=-np.inf):
        info = dict(terms_evaluated=count,
                    relative_truncation_bound=float(np.exp(log_bound)),
                    log_error_bound=float(np.logaddexp(0.0, log_bound)))
        return (float(value), info) if return_info else float(value)

    if not np.isfinite(x) or x != np.floor(x) or not 0 <= x <= n:
        return result(-np.inf)
    x = int(x)
    if n == 0:
        return result(0.0)
    if D == 0 or D == 1:
        return result(0.0 if x == n * D else -np.inf)

    log_end = np.logaddexp(B*np.log1p(-D) if x == 0 else -np.inf,
                          B*np.log(D) if x == n else -np.inf)
    if B == 1:
        return result(log_end)
    log_odds = np.log(D) - np.log1p(-D)

    def log_ratio(c):
        # log(w(c+1)/w(c)); only evaluated for 1 <= c <= B-2.
        return (np.log(B-c) - np.log(c+1) + log_odds
                + np.log1p(x/c) - np.log1p((n-x)/(B-c-1)))

    # Binary search for an interior mode, O(log B).
    left, right = 1, B-1
    while left < right:
        mid = (left + right)//2
        if log_ratio(mid) > 0:
            left = mid + 1
        else:
            right = mid
    mode = left
    # One absolute evaluation; other terms use stable relative recurrences.
    anchor = float(binom.logpmf(mode, B, D)
                   + betabinom.logpmf(x, n, mode, B-mode))

    def tail_bound(log_boundary, log_outward_ratio):
        if log_outward_ratio >= 0:
            return np.inf
        return (log_boundary + log_outward_ratio
                - np.log(-np.expm1(log_outward_ratio)))

    width = 32
    evaluated = 0
    while True:
        L, U = max(1, mode-width), min(B-1, mode+width)
        lc = np.arange(mode-1, L-1, -1, dtype=float)
        rc = np.arange(mode, U, dtype=float)
        left_logs = -np.cumsum(log_ratio(lc))
        right_logs = np.cumsum(log_ratio(rc))
        relative_logs = np.concatenate(([0.0], left_logs, right_logs))
        evaluated += len(relative_logs)
        log_sum = float(np.logaddexp(anchor + logsumexp(relative_logs), log_end))
        tl = (-np.inf if L == 1 else
              tail_bound(left_logs[-1], -log_ratio(float(L-1))))
        tr = (-np.inf if U == B-1 else
              tail_bound(right_logs[-1], log_ratio(float(U))))
        log_bound = float(anchor + np.logaddexp(tl, tr) - log_sum)
        if log_bound <= np.log(rtol):
            return result(log_sum, evaluated, log_bound)
        if not np.isfinite(log_sum):
            raise FloatingPointError('Nonfinite probability; check parameter scale.')
        width *= 2


def h_B(x, n, D, B, *, rtol=1e-8):
    """Probability; use log_h_B directly for likelihoods to avoid underflow."""
    return float(np.exp(log_h_B(x, n, D, B, rtol=rtol)))


def _loglik_feature(q, x_vec, n_vec, Nb_vec, *,
                    q_bounds=(0.0, 1.0), rtol=1e-8):
    """Sum log h_{B_m}(x_m; n_m,q) over independent hosts.

    Keeps the previous function's clipping behavior. Remove alpha0 from calls.
    Total truncation error in log likelihood <= M*log1p(rtol).
    """
    arrays = [np.asarray(v) for v in (x_vec, n_vec, Nb_vec)]
    if any(v.ndim != 1 for v in arrays) or not (
            arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError('Input arrays must be one-dimensional and equally sized.')
    lo, hi = q_bounds
    if not 0 <= lo <= hi <= 1 or not np.isfinite(q):
        raise ValueError('Invalid q or q_bounds.')
    q = float(np.clip(q, lo, hi))
    return float(sum(log_h_B(x, n, q, B, rtol=rtol)
                     for x, n, B in zip(*arrays)))


_MIN_PVAL = float(np.nextafter(0.0, 1.0))  # smallest positive float64


def _fit_q_mle(x_vec: np.ndarray, n_vec: np.ndarray, Nb_vec: np.ndarray, *, alpha0: float= 1e-12, q_bounds: Tuple[float, float],
              )-> Tuple[float, float]:
    """
    MLE for q for one feature. Returns (q_hat, loglik_at_hat).
    """
    lo, hi = q_bounds

    def nll(q: float) -> float:
        return -_loglik_feature(q, x_vec, n_vec, Nb_vec, q_bounds=q_bounds, rtol=1e-8)

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
        ll0 = _loglik_feature(p0, x_vec, n_tot, Nb_vec, q_bounds=q_bounds, rtol=1e-8)
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

    res = pd.DataFrame({"feature": list(feature_ids), "p_donor": p_donor, "q_hat_recipient_center": q_hat, "LR": LR,
                        "pval": pval, "qval_FDR": qval, "reject_FDR": reject,})

    direction = pd.Series([None]*len(res), dtype="object")
    mask = np.isfinite(res["q_hat_recipient_center"].to_numpy(dtype=float))
    direction.loc[mask] = np.where(res.loc[mask, "q_hat_recipient_center"] > res.loc[mask, "p_donor"], "up_in_recipient", "down_in_recipient",)
    res["direction"] = direction
    return res.sort_values(["pval", "LR"], ascending=[True, False]).reset_index(drop=True)

import numpy as np
from scipy.special import gammaln
from functools import lru_cache
from scipy.special import logsumexp

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

import ctypes
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from scipy.special import digamma

_FOUNDER_SADDLE_CPP = '#include <cstdint>\n#include <cmath>\n#include <vector>\n#include <algorithm>\n#include <limits>\n\n// F_x(t)=sum_{c>=0} t^c (c)_x/c!. Moments of its normalized coefficients.\n// All moments accumulated about the mode to avoid cancellation.\nstatic void moments(long x,double t,double *o){\n if(x==0){o[0]=t;o[1]=t;o[2]=t;o[3]=t;o[4]=t;return;}\n if(t<=0){o[0]=-INFINITY;for(int k=1;k<5;k++)o[k]=0;return;}\n double root=(t-1+std::sqrt((t-1)*(t-1)+4*t*x))/2;\n long mode=std::max(1L,(long)std::ceil(root));\n double a[5]={1,0,0,0,0},w=1;\n for(long c=mode;c>1;--c){\n   w*=double(c)*(c-1)/(t*(c+x-1));double d=double(c-1-mode),p=1;\n   for(int k=0;k<5;k++){a[k]+=w*p;p*=d;}\n   if(w<1e-16*a[0])break;\n }\n w=1;\n for(long c=mode;c<mode+10000000;++c){\n   w*=t*(c+x)/(double(c)*(c+1));double d=double(c+1-mode),p=1;\n   for(int k=0;k<5;k++){a[k]+=w*p;p*=d;}\n   if(w<1e-16*a[0])break;\n }\n double m=a[1]/a[0],v=a[2]/a[0]-m*m;\n o[0]=mode*std::log(t)+std::lgamma(mode+x)-std::lgamma(mode)-std::lgamma(mode+1)+std::log(a[0]);\n o[1]=mode+m;o[2]=v;\n o[3]=a[3]/a[0]-3*m*a[2]/a[0]+2*m*m*m;\n o[4]=a[4]/a[0]-4*m*a[3]/a[0]+6*m*m*a[2]/a[0]-3*std::pow(m,4)-3*v*v;\n}\n\nextern "C" int saddle(int K,const int64_t *x,const double *q,double B,\n                     double *out,double *g){\n double n=0;int s=0;for(int j=0;j<K;j++){n+=x[j];s+=(x[j]>0);if(x[j]>0&&q[j]<=0)return 2;}\n if(B<=s)return 3;\n double z=std::log(B*B/(B+n)),mu=0,V=0,C3=0,C4=0,L=0;\n std::vector<double> stats(K*5);\n int it;\n for(it=0;it<60;it++){\n   mu=V=C3=C4=L=0;\n   for(int j=0;j<K;j++){\n     moments(x[j],std::exp(z)*q[j],&stats[j*5]);\n     L+=stats[j*5];mu+=stats[j*5+1];V+=stats[j*5+2];\n     C3+=stats[j*5+3];C4+=stats[j*5+4];\n   }\n   if(std::abs(mu-B)<1e-9*std::max(1.,B))break;\n   if(V<=0||!std::isfinite(L))return 4;\n   z-=std::max(-3.,std::min(3.,(mu-B)/V));\n }\n if(it==60)return 5;\n out[0]=std::lgamma(n+1)+std::lgamma(B+1)+std::lgamma(B)-std::lgamma(B+n)+L-B*z-.5*std::log(2*3.141592653589793238462643383279502884*V);\n for(int j=0;j<K;j++){\n   out[0]-=std::lgamma(x[j]+1);\n   g[j]=stats[j*5+1]-.5*(stats[j*5+3]/V-C3*stats[j*5+2]/(V*V));\n }\n out[1]=z;out[2]=V;out[3]=C3;out[4]=C4/(8*V*V)-5*C3*C3/(24*V*V*V);\n return 0;\n}\n\n'
_saddle_lib = None

def core():
    global _saddle_lib
    if _saddle_lib is None:
        compiler = shutil.which('g++') or shutil.which('clang++')
        if compiler is None:
            raise RuntimeError('Install g++ or clang++ (a C++17 compiler), then restart the kernel.')
        build = Path.cwd() / '.founder_saddle_build'
        build.mkdir(exist_ok=True)
        token = hashlib.sha256(_FOUNDER_SADDLE_CPP.encode()).hexdigest()[:16]
        source = build / ('founder_saddle_' + token + '.cpp')
        binary = build / ('founder_saddle_' + token + ('.dll' if sys.platform == 'win32' else '.so'))
        if not binary.exists():
            source.write_text(_FOUNDER_SADDLE_CPP)
            flags = ['-O3', '-shared', '-std=c++17']
            if sys.platform != 'win32': flags.append('-fPIC')
            subprocess.run([compiler, *flags, str(source), '-o', str(binary)], check=True)
        lib = ctypes.CDLL(str(binary.resolve()))
        ar = np.ctypeslib.ndpointer
        lib.saddle.argtypes = [ctypes.c_int, ar(np.int64, flags='C_CONTIGUOUS'),
            ar(np.float64, flags='C_CONTIGUOUS'), ctypes.c_double,
            ar(np.float64, flags='C_CONTIGUOUS'), ar(np.float64, flags='C_CONTIGUOUS')]
        lib.saddle.restype = ctypes.c_int
        lib.fast_ll.argtypes = [ctypes.c_int] + [ar(np.int64, flags='C_CONTIGUOUS')]*3 + [ctypes.c_double]
        lib.fast_ll.restype = ctypes.c_double
        lib.fast_lrt.argtypes = [ctypes.c_int] + [ar(np.int64, flags='C_CONTIGUOUS')]*3 + [ctypes.c_double]*3 + [ar(np.float64, flags='C_CONTIGUOUS')]
        lib.fast_lrt.restype = ctypes.c_int
        _saddle_lib = lib
    return _saddle_lib

def integer_counts(a):
    a=np.asarray(a,dtype=float)
    if not np.isfinite(a).all() or np.any(a<0) or np.any(a!=np.floor(a)):
        raise ValueError('Counts must be finite nonnegative integers; missing counts are not zero.')
    return np.ascontiguousarray(a,dtype=np.int64)

def multinomial_logpmf(x,q):
    x=integer_counts(x);q=np.asarray(q,float)
    if np.any((x>0)&(q<=0)):return -np.inf
    return float(gammaln(x.sum()+1)-gammaln(x+1).sum()+np.dot(x[x>0],np.log(q[x>0])))

def saddle_logpmf(x,b,q):
    x=np.ascontiguousarray(x,np.int64);q=np.ascontiguousarray(q,float)
    out=np.empty(5);g=np.empty(len(q))
    err=core().saddle(len(q),x,q,float(b),out,g)
    if err:raise ArithmeticError(f'Saddlepoint evaluator status {err}; B={b}, observed support={np.count_nonzero(x)}')
    gb=digamma(b+1)+digamma(b)-digamma(b+x.sum())-out[1]-.5*out[3]/out[2]**2
    return out[0],g,float(gb),out[4]

def exact_joint_low_excess(x,b,q):
    """Exact coefficient convolution when B - observed richness is small."""
    x=integer_counts(x);q=np.asarray(q,float);s=np.count_nonzero(x);e=int(b)-s
    if e<0:return -np.inf
    if np.any((x>0)&(q<=0)):return -np.inf
    poly=np.zeros(e+1);poly[0]=1.;scale=0.
    for xx,qq in zip(x,q):
        a=np.ones(e+1)
        for r in range(e):
            a[r+1]=a[r]*qq*((r+1+xx)/((r+1)*(r+2)) if xx>0 else 1/(r+1))
        poly=np.convolve(poly,a)[:e+1]
        norm=poly.max();poly/=norm;scale+=np.log(norm)
    return float(gammaln(x.sum()+1)+gammaln(b+1)+gammaln(b)-gammaln(b+x.sum())+np.log(q[x>0]).sum()+np.log(poly[e])+scale)

def _saddle_bottleneck_mle_from_p(counts_t1,p,nb_max=1e8):
    """Single-host neutral founder-model B estimate, with numerical metadata.

    Exact convolution is used for low excess founders; otherwise coefficient
    saddlepoint approximation. The multinomial B=infinity limit is compared.
    A finite cap that still improves is flagged rather than silently reported.
    Selection can confound this globally neutral fit. For BN comparisons use
    the jointly fitted q diagnostic, not this null estimate as known truth.
    """
    x=integer_counts(counts_t1);p=np.asarray(p,float);s=int(np.count_nonzero(x))
    if x.ndim!=1 or p.shape!=x.shape or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1.,rtol=0,atol=1e-10):
        raise ValueError('A normalized probability vector matching one-dimensional recipient counts is required')
    if not np.isfinite(nb_max) or nb_max<s+12:raise ValueError('Finite search cap must exceed observed richness by at least 12')
    if s==0:raise ValueError('Empty recipient')
    if np.any((x>0)&(p<=0)):raise ValueError('Positive recipient count with donor frequency zero')
    def ll(b):
        bi=int(round(b))
        if bi-s<=32:return exact_joint_low_excess(x,bi,p)
        v,_,_,cor=saddle_logpmf(x,bi,p);return v+cor
    grid=np.unique(np.r_[np.arange(s,s+12),np.rint(np.geomspace(s+12,nb_max,70))]).astype(int)
    vals=np.array([ll(b) for b in grid]);i=int(vals.argmax());b0=grid[i]
    if 0<i<len(grid)-1:
        opt=minimize_scalar(lambda z:-ll(np.exp(z)),bounds=(np.log(grid[i-1]),np.log(grid[i+1])),method='bounded',options={'xatol':1e-7})
        b0=int(round(np.exp(opt.x)))
    candidates=np.arange(max(s,b0-2),b0+3);b=int(max(candidates,key=ll));best=ll(b)
    li=multinomial_logpmf(x,p)
    if li>=best-1e-7:return {'B_hat':np.inf,'loglik':li,'status':'infinite_boundary','richness':s,'approximation':'coefficient_saddlepoint_or_exact_low_excess'}
    if b>=.999*nb_max:return {'B_hat':np.nan,'loglik':best,'status':'unresolved_search_cap','richness':s,'approximation':'coefficient_saddlepoint'}
    return {'B_hat':b,'loglik':best,'status':'finite','richness':s,'approximation':'exact_low_excess' if b-s<=32 else 'coefficient_saddlepoint'}

def bottleneck_mle_from_p(counts_t1, p, nb_max=5_000_000,
                          n_grid=220, refine=220, alpha0=1e-12):
    """Approximate founder-mixture B fit; compatibility wrapper.

    n_grid/refine/alpha0 are retained for call compatibility but unused.
    The supplied approximation uses its own 70-point grid and local search.
    Infinity denotes the multinomial limit. An unresolved cap raises.
    """
    result = _saddle_bottleneck_mle_from_p(counts_t1, p, nb_max=nb_max)
    if result['status'] == 'unresolved_search_cap':
        raise RuntimeError('Unresolved bottleneck search cap; increase nb_max.')
    return float(result['B_hat'])


def bottleneck_from_two_timepoints(counts_t0, counts_t1,
                                   donor_pseudocount=1e-6, alpha0=1e-12,
                                   nb_max=5_000_000, n_grid=220, refine=220,
                                   return_p=False):
    p = donor_frequencies(counts_t0, pseudocount=donor_pseudocount)
    b = bottleneck_mle_from_p(counts_t1, p, nb_max=nb_max)
    return (b, p) if return_p else b

_FOUNDER_SADDLE_CPP += '\n// Bounded Brent minimization follows the method used by SciPy\'s\n// _minimize_scalar_bounded; implementation below retains its default tolerance.\n#include <cstdint>\n#include <cmath>\n#include <algorithm>\n#include <limits>\nstatic double addlog(double a,double b){if(a==-INFINITY)return b;if(b==-INFINITY)return a;double m=std::max(a,b);return m+std::log1p(std::exp(std::min(a,b)-m));}\nstatic double lr_ratio(int64_t c,int64_t x,int64_t n,int64_t B,double odds){return std::log(double(B-c))-std::log(double(c+1))+odds+std::log1p(double(x)/c)-std::log1p(double(n-x)/(B-c-1));}\nstatic double fast_logh(int64_t x,int64_t n,int64_t B,double p){\n if(n==0)return 0;\n if(p==0||p==1)return (x==n*p)?0:-INFINITY;\n if(B<0)return std::lgamma(n+1.)-std::lgamma(x+1.)-std::lgamma(n-x+1.)+x*std::log(p)+(n-x)*std::log1p(-p);\n double ends=addlog(x==0?B*std::log1p(-p):-INFINITY,x==n?B*std::log(p):-INFINITY);\n if(B==1)return ends;\n double odds=std::log(p)-std::log1p(-p);\n int64_t l=1,u=B-1;\n while(l<u){int64_t mid=l+(u-l)/2;if(lr_ratio(mid,x,n,B,odds)>0)l=mid+1;else u=mid;}\n int64_t mode=l;\n // Long-double gamma differences reduce cancellation at large B.\n auto G=[](long double v){return std::lgamma(v);};\n double anchor=double(G(B+1.L)-G(mode+1.L)-G(B-mode+1.L)+mode*std::log((long double)p)+(B-mode)*std::log1p(-(long double)p)\n +G(n+1.L)-G(x+1.L)-G(n-x+1.L)+G(mode+x+0.L)-G(mode+0.L)+G(B-mode+n-x+0.L)-G(B-mode+0.L)+G(B+0.L)-G(B+n+0.L));\n double sum=1,term=1;\n for(int64_t c=mode;c>1;){term*=std::exp(-lr_ratio(c-1,x,n,B,odds));sum+=term;--c;\n  if(c>1){double r=std::exp(-lr_ratio(c-1,x,n,B,odds));if(r<1&&term*r/(1-r)<=5e-9*sum)break;}}\n term=1;\n for(int64_t c=mode;c<B-1;){term*=std::exp(lr_ratio(c,x,n,B,odds));sum+=term;++c;\n  if(c<B-1){double r=std::exp(lr_ratio(c,x,n,B,odds));if(r<1&&term*r/(1-r)<=5e-9*sum)break;}}\n return addlog(anchor+std::log(sum),ends);\n}\nextern "C" double fast_ll(int M,const int64_t*x,const int64_t*n,const int64_t*b,double q){double v=0;for(int m=0;m<M;m++)v+=fast_logh(x[m],n[m],b[m],q);return v;}\nextern "C" int fast_lrt(int M,const int64_t*x,const int64_t*n,const int64_t*b,double p0,double low,double high,double*out){\n auto f=[&](double q){return -fast_ll(M,x,n,b,q);};\n double a=low,bb=high,g=.5*(3-std::sqrt(5.)),xf=a+g*(bb-a),w=xf,v=xf;\n double fx=f(xf),fw=fx,fv=fx,e=0,d=0;\n for(int it=0;it<500;it++){\n  double mid=.5*(a+bb),tol=std::sqrt(2.2e-16)*std::abs(xf)+1e-5/3.,tol2=2*tol;\n  if(std::abs(xf-mid)<=tol2-.5*(bb-a)){double null=f(p0);out[0]=std::max(0.,2*(null-fx));out[1]=xf;out[2]=-fx;return std::isfinite(null)&&std::isfinite(fx)?0:2;}\n  bool golden=true;\n  if(std::abs(e)>tol){double r=(xf-w)*(fx-fv),q=(xf-v)*(fx-fw),p=(xf-v)*q-(xf-w)*r;q=2*(q-r);if(q>0)p=-p;q=std::abs(q);r=e;e=d;\n   if(std::abs(p)<std::abs(.5*q*r)&&p>q*(a-xf)&&p<q*(bb-xf)){d=p/q;golden=false;double z=xf+d;if(z-a<tol2||bb-z<tol2)d=(mid>=xf?tol:-tol);}}\n  if(golden){e=xf>=mid?a-xf:bb-xf;d=g*e;}\n  double z=xf+(d>=0?1:-1)*std::max(std::abs(d),tol),fz=f(z);\n  if(fz<=fx){if(z>=xf)a=xf;else bb=xf;v=w;fv=fw;w=xf;fw=fx;xf=z;fx=fz;}\n  else{if(z<xf)a=z;else bb=z;if(fz<=fw||w==xf){v=w;fv=fw;w=z;fw=fz;}else if(fz<=fv||v==xf||v==w){v=z;fv=fz;}}\n }\n return 1;\n}\n'


def _cpp_lrt(x, n, B, p0, q_bounds):
    x = np.ascontiguousarray(x, dtype=np.int64)
    n = np.ascontiguousarray(n, dtype=np.int64)
    B = np.asarray(B, dtype=float)
    finite = np.isfinite(B)
    bs = np.full(B.shape, -1, dtype=np.int64)
    bs[finite] = B[finite].astype(np.int64)
    out = np.empty(3)
    status = core().fast_lrt(len(x), x, n, bs, float(p0),
                             float(q_bounds[0]), float(q_bounds[1]), out)
    if status:
        raise RuntimeError(f'C++ LR evaluation failed: status {status}')
    return float(out[0]), float(out[1])


from concurrent.futures import ThreadPoolExecutor
import os

import time
import warnings
from scipy.stats import beta as beta_distribution


def _validated_counts(values, ndim, name):
    a = np.asarray(values, dtype=float)
    if a.ndim != ndim or not a.size or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a nonempty, finite {ndim}D array.')
    if np.any(a < 0) or np.any(a != np.floor(a)) or np.any(a > 2**53):
        raise ValueError(f'{name} must contain nonnegative integer counts <= 2**53.')
    return a.astype(np.int64)


def _simulate_founder_counts(p, totals, bottlenecks, rng):
    """Joint simulator matching dm_loglik and the marginals log_h_B.

    C_m ~ Multinomial(B_m, p).
    On C_m > 0, P_m ~ Dirichlet(C_m); elsewhere P_m = 0.
    X_m ~ Multinomial(n_m, P_m).
    Zero-founder components stay zero; no Dirichlet pseudocount is added.
    """
    X = np.zeros((len(totals), len(p)), dtype=np.int64)
    for m, (n, B) in enumerate(zip(totals, bottlenecks)):
        if np.isposinf(B):
            X[m] = rng.multinomial(int(n), p)
            continue
        founders = rng.multinomial(int(B), p)
        active = founders > 0
        probs = rng.dirichlet(founders[active].astype(float))
        X[m, active] = rng.multinomial(int(n), probs)
    return X


def neutrality_lrt_bootstrap(
    donor_counts, recipient_counts, *, n_boot=999,
    reestimate_bottleneck=False, bottleneck=None, seed=12345,
    feature_ids=None, pseudocount=1e-6, min_p=1e-8,
    q_bounds=None, fdr_alpha=0.05,
    fdr_method='fdr_bh', nb_max=5_000_000, n_grid=220, refine=220,
    progress_every=10, store_statistics=False, n_jobs=2, lr_backend=None,
):
    """Two-sided LR bootstrap requiring fitted/observed direction agreement.

    Recipient matrix: hosts x variants. Use all variants for totals and B.
    reestimate_bottleneck=True fits B anew in each bootstrap replicate.
    Otherwise observed or supplied B is fixed. Generation always uses
    observed B. q is fitted anew for every variant in every replicate.
    Returned LR is ungated; agreement_LR is the actual test statistic.
    Direction is observed frequency direction; direction_agreement records
    whether q supports it. Discordant/equal directions have T=0 and p=1.
    mc_tail_ci95 columns quantify Monte Carlo tail error only.
    Approximate complete-null bootstrap; mixture-null FDR not established.
    """
    q_bounds = (1e-12, 1-1e-12) if q_bounds is None else q_bounds
    lr_backend = 'cpp' if lr_backend is None else lr_backend
    if lr_backend not in ('cpp', 'python'):
        raise ValueError("lr_backend must be cpp or python")
    lo, hi = q_bounds
    if not 0 < lo < hi < 1:
        raise ValueError('Require 0 < q_bounds[0] < q_bounds[1] < 1.')
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs == 0 or n_jobs < -1:
        raise ValueError('n_jobs must be -1 or a positive integer.')
    if n_jobs == -1:
        n_jobs = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    # Compile once on the main thread, before concurrent access.
    if bottleneck is None or lr_backend == 'cpp':
        core()
    if isinstance(n_boot, bool) or int(n_boot) != n_boot or n_boot < 1:
        raise ValueError('n_boot must be a positive integer.')
    n_boot = int(n_boot)
    if not isinstance(reestimate_bottleneck, (bool, np.bool_)):
        raise ValueError('reestimate_bottleneck must be True or False.')
    for name, value, lower in [('nb_max', nb_max, 1), ('n_grid', n_grid, 2), ('refine', refine, 2)]:
        if not np.isfinite(value) or int(value) != value or value < lower:
            raise ValueError(f'{name} must be an integer >= {lower}.')
    nb_max, n_grid, refine = int(nb_max), int(n_grid), int(refine)
    if not np.isfinite(pseudocount) or pseudocount < 0:
        raise ValueError('pseudocount must be finite and nonnegative.')
    if not 0 < fdr_alpha < 1 or not 0 <= min_p < 0.5:
        raise ValueError('Invalid fdr_alpha or min_p.')
    if int(progress_every) != progress_every or progress_every < 0:
        raise ValueError('progress_every must be a nonnegative integer.')

    # Validate labels rather than silently comparing misaligned variants.
    if isinstance(recipient_counts, pd.DataFrame):
        if not recipient_counts.columns.is_unique:
            raise ValueError('Recipient feature columns must be unique.')
        ids = list(recipient_counts.columns)
        hosts = list(recipient_counts.index)
        if isinstance(donor_counts, pd.Series):
            if not donor_counts.index.is_unique or set(donor_counts.index) != set(ids):
                raise ValueError('Donor labels must match recipient columns exactly.')
            donor_counts = donor_counts.reindex(ids)
    else:
        ids = list(donor_counts.index) if isinstance(donor_counts, pd.Series) else None
        hosts = None
    X = _validated_counts(recipient_counts, 2, 'recipient_counts')
    donor = _validated_counts(donor_counts, 1, 'donor_counts')
    M, K = X.shape
    if donor.sum(dtype=float) > 2**53:
        raise ValueError('Donor sequencing total exceeds supported range.')
    if len(donor) != K or K < 2 or donor.sum() <= 0:
        raise ValueError('Require matching donor counts and at least two variants.')
    if np.any(X.sum(axis=1, dtype=float) > 2**53):
        raise ValueError('Host sequencing totals exceed supported range.')
    totals = X.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError('Every host must have a positive sequencing total.')
    ids = list(feature_ids) if feature_ids is not None else (ids if ids is not None else list(range(K)))
    if len(ids) != K or not pd.Index(ids).is_unique:
        raise ValueError('feature_ids must be unique and match the number of variants.')
    hosts = hosts if hosts is not None else list(range(M))
    p = donor_frequencies(donor, pseudocount=pseudocount)
    tested = (p >= min_p) & (p <= 1 - min_p) & (p > 0) & (p < 1)
    if np.any(tested & ((p < lo) | (p > hi))):
        raise ValueError('q_bounds must contain all tested donor frequencies.')
    if not tested.any():
        raise ValueError('No variants pass min_p.')
    if np.any((X > 0) & (p[None, :] == 0)):
        raise ValueError('Positive recipient count has zero donor probability; use a positive pseudocount.')
    if bottleneck is not None and reestimate_bottleneck:
        raise ValueError('With refitting, omit bottleneck so observed and simulated fits use the same estimator.')

    def fit_bottlenecks(counts):
        fits = []
        for m in range(M):
            b = bottleneck_mle_from_p(counts[m], p, nb_max=nb_max, n_grid=n_grid, refine=refine)
            if np.isnan(b) or b < 1 or (np.isfinite(b) and b != np.floor(b)):
                raise RuntimeError(f'Bottleneck fit failed for host {hosts[m]!r}.')
            fits.append(float(b))
        return np.asarray(fits, dtype=float)

    def statistics(counts, Bs):
        mean_frequency = np.mean(counts / totals[:, None], axis=0)
        lr = np.full(K, np.nan)
        qhat = np.full(K, np.nan)
        for j in np.flatnonzero(tested):
            if lr_backend == 'cpp':
                lr[j], qhat[j] = _cpp_lrt(counts[:, j], totals, Bs, p[j], q_bounds)
            else:
                ll0 = _loglik_feature(p[j], counts[:, j], totals, Bs, q_bounds=q_bounds, rtol=1e-8)
                qhat[j], ll1 = _fit_q_mle(counts[:, j], totals, Bs, q_bounds=q_bounds)
                if not np.isfinite(ll0) or not np.isfinite(ll1):
                    raise FloatingPointError('Nonfinite likelihood.')
                lr[j] = max(0., 2*(ll1-ll0))
        agree = tested & (((qhat > p) & (mean_frequency > p)) |
                          ((qhat < p) & (mean_frequency < p)))
        statistic = np.where(tested, np.where(agree, lr, 0.), np.nan)
        return statistic, mean_frequency, qhat, lr, agree

    start = time.perf_counter()
    if progress_every:
        print('Fitting observed bottlenecks and computing observed statistics...', flush=True)
    if bottleneck is None:
        B_observed = fit_bottlenecks(X)
    else:
        b = _as_1d_float_array(bottleneck, M)
        if np.isnan(b).any() or np.any(b < 1) or np.any(b[np.isfinite(b)] != np.floor(b[np.isfinite(b)])) or np.any(b[np.isfinite(b)] > 2**53):
            raise ValueError('bottleneck must contain positive integers <= 2**53 or positive infinity.')
        B_observed = b.astype(float)
    T_observed, mean_frequency, qhat, raw_lr, agreement = statistics(X, B_observed)
    observed_seconds = time.perf_counter() - start
    rng = np.random.default_rng(seed)
    exceedances = np.zeros(K, dtype=np.int64)
    B_boot = np.empty((n_boot, M), dtype=float)
    T_boot = np.full((n_boot, K), np.nan) if store_statistics else None
    simulation_seconds = 0.0
    fit_times = np.zeros(n_boot)
    statistic_times = np.zeros(n_boot)

    def evaluate(task):
        r, simulated = task
        try:
            t0 = time.perf_counter()
            B_rep = fit_bottlenecks(simulated) if reestimate_bottleneck else B_observed
            t1 = time.perf_counter()
            T_rep, _, _, _, _ = statistics(simulated, B_rep)
            return r, B_rep, T_rep, t1 - t0, time.perf_counter() - t1
        except Exception as exc:
            raise RuntimeError(f'Bootstrap replicate {r + 1} failed; no p-values returned.') from exc

    def collect(item):
        r, B_rep, T_rep, fit_time, statistic_time = item
        B_boot[r] = B_rep
        fit_times[r], statistic_times[r] = fit_time, statistic_time
        # Include numerical ties conservatively.
        tol = 1e-12 * np.maximum(1.0, np.abs(T_observed[tested]))
        exceedances[tested] += T_rep[tested] >= T_observed[tested] - tol
        if T_boot is not None:
            T_boot[r] = T_rep
        if progress_every and ((r + 1) % progress_every == 0 or r == 0 or r + 1 == n_boot):
            print(f'Bootstrap {r + 1}/{n_boot}; elapsed {time.perf_counter() - start:.1f} s', flush=True)

    # Simulations use the original RNG sequentially on the main thread.
    # Results are collected in replicate order, independently of worker count.
    # Bounded batches avoid retaining every simulated count matrix in memory.
    workers = min(n_jobs, n_boot)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for first in range(0, n_boot, 2 * workers):
            t0 = time.perf_counter()
            tasks = [(r, _simulate_founder_counts(p, totals, B_observed, rng))
                     for r in range(first, min(n_boot, first + 2 * workers))]
            simulation_seconds += time.perf_counter() - t0
            items = map(evaluate, tasks) if workers == 1 else executor.map(evaluate, tasks)
            for item in items:
                collect(item)

    pval = np.full(K, np.nan)
    raw_tail = np.full(K, np.nan)
    raw_tail[tested] = (exceedances[tested] + 1) / (n_boot + 1)
    pval[tested] = raw_tail[tested]
    qval = np.full(K, np.nan)
    reject = np.zeros(K, dtype=bool)
    reject[tested], qval[tested], _, _ = multipletests(pval[tested], alpha=fdr_alpha, method=fdr_method)
    # Pointwise 95% binomial intervals for the Monte Carlo exceedance probability.
    # These quantify simulation error, NOT uncertainty in the fitted null model.
    ci_lo, ci_hi = np.full(K, np.nan), np.full(K, np.nan)
    k = exceedances[tested]
    lower = np.zeros(len(k)); upper = np.ones(len(k))
    positive = k > 0; below = k < n_boot
    lower[positive] = beta_distribution.ppf(0.025, k[positive], n_boot - k[positive] + 1)
    upper[below] = beta_distribution.ppf(0.975, k[below] + 1, n_boot - k[below])
    ci_lo[tested], ci_hi[tested] = lower, upper
    direction = np.full(K, None, dtype=object)
    direction[tested] = np.where(mean_frequency[tested] > p[tested], 'up_in_recipient',
        np.where(mean_frequency[tested] < p[tested], 'down_in_recipient', 'unchanged'))
    hits = np.full(K, np.nan); hits[tested] = exceedances[tested]
    results = pd.DataFrame(dict(feature=ids, p_donor=p, mean_recipient_frequency=mean_frequency,
        frequency_ratio=np.divide(mean_frequency, p, out=np.full(K, np.nan), where=p > 0),
        q_hat_recipient_center=qhat, LR=raw_lr, direction_agreement=agreement,
        tested=tested, agreement_LR=T_observed, pval=pval, qval_FDR=qval, reject_FDR=reject, direction=direction,
        bootstrap_raw_tail_pvalue=raw_tail, bootstrap_exceedances=hits, bootstrap_replicates=n_boot,
        mc_tail_ci95_low=ci_lo, mc_tail_ci95_high=ci_hi))
    diagnostics = dict(
        observed_bottlenecks=pd.Series(B_observed, index=hosts, name='B_hat'),
        bootstrap_bottlenecks=pd.DataFrame(B_boot, columns=hosts),
        bootstrap_statistics=None if T_boot is None else pd.DataFrame(T_boot, columns=ids),
        observed_statistics=pd.Series(T_observed, index=ids, name='agreement_LR'),
        reestimate_bottleneck=bool(reestimate_bottleneck), seed=seed,
        statistic='direction_agreement_likelihood_ratio', lr_backend=lr_backend, alternative='two-sided', n_jobs=workers, simulation_seconds=simulation_seconds,
        bootstrap_bottleneck_worker_seconds=fit_times,
        bootstrap_statistic_worker_seconds=statistic_times,
        null_model='all variants neutral; donor frequencies and read totals fixed',
        minimum_pvalue=1 / (n_boot + 1), observed_fit_seconds=observed_seconds,
        elapsed_seconds=time.perf_counter() - start,
        observed_at_nb_max=(B_observed == nb_max),
        bottleneck_method='corrected_coefficient_saddlepoint_or_exact_low_excess',
        observed_infinite_boundary=np.isposinf(B_observed),
        bootstrap_fraction_infinite=np.mean(np.isposinf(B_boot), axis=0),
        bootstrap_fraction_at_nb_max=np.mean(B_boot == nb_max, axis=0),
    )
    return results.sort_values(['pval', 'agreement_LR'], ascending=[True, False]).reset_index(drop=True), diagnostics


# Preferred entry point. The legacy LRT name also runs the two-sided test.
neutrality_bootstrap = neutrality_lrt_bootstrap


def neutrality_depletion_bootstrap(*args, **kwargs):
    """Deprecated name: now runs the TWO-SIDED test; use neutrality_bootstrap."""
    warnings.warn('neutrality_depletion_bootstrap now runs a two-sided test; '
                  'use neutrality_bootstrap.', FutureWarning, stacklevel=2)
    return neutrality_bootstrap(*args, **kwargs)
