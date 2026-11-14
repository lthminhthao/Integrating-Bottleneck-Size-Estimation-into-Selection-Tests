"""Approximate bottleneck estimation extracted from neutrality_test_bootstrap.

Requires NumPy, SciPy, and g++ or clang++ with C++17 support. Embedded C++
is compiled automatically on first saddlepoint evaluation.

Uses corrected coefficient saddlepoint approximation for B - richness > 32,
exact low-excess convolution otherwise, and compares the B=infinity limit.
The donor pseudocount (1e-6), search cap (5_000_000), and estimator are unchanged.
The saddlepoint correction is not a certified error bound.

Usage:
    from bottleneck_function import bottleneck_from_two_timepoints
    B_hat = bottleneck_from_two_timepoints(donor_counts, recipient_counts)

Inputs are one-dimensional count vectors with matching variant order; call
once per recipient host. Returns a scalar float (integer-valued or np.inf).
np.inf denotes the multinomial limit. An unresolved finite search cap raises.
No neutrality testing or bootstrap is performed in this module.
"""

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize_scalar

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
