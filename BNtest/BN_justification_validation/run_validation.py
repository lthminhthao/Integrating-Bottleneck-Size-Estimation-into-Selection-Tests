"""Null calibration and power of the supplied BN implementation; resumable.
Exactly five hosts and 200 inner bootstrap replicates in every run.
"""
from pathlib import Path
import hashlib, json, time, argparse, platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import beta, t
import neutrality_test_bootstrap as bn

N_BOOT = 200
N_HOSTS = 5
ALPHAS = (0.01, 0.05, 0.10)
# Settings chosen before examining results; all cases must be reported.
SCENARIOS = [
    dict(name='null_small', K=50, profile='uniform', B=[20]*5, depths=[10000]*5, alternative=False),
    dict(name='null_moderate', K=100, profile='unequal', B=[200]*5, depths=[10000]*5, alternative=False),
    dict(name='null_large', K=100, profile='unequal', B=[2000]*5, depths=[10000]*5, alternative=False),
    dict(name='null_heterogeneous', K=100, profile='unequal', B=[20,50,200,800,2000], depths=[2000,5000,10000,20000,50000], alternative=False),
    dict(name='power_moderate', K=100, profile='unequal', B=[200]*5, depths=[10000]*5, alternative=True),
    dict(name='power_large', K=100, profile='unequal', B=[2000]*5, depths=[10000]*5, alternative=True),
]

def design(case):
    K = case['K']
    assert 2 <= K <= 100 and len(case['B']) == len(case['depths']) == N_HOSTS
    # Positive, fixed donor counts. In unequal cases adjacent pairs have equal mass.
    donor = (np.full(K,1000,dtype=int) if case['profile']=='uniform' else
             np.repeat(np.rint(np.geomspace(100,10000,K//2)).astype(int),2))
    D = donor / donor.sum()
    q = D.copy()
    if case['alternative']:
        # Ten disjoint pairs across the abundance range: 10 depleted, 10 enriched.
        # Transfer 75% within each pair: q/D = 0.25 and 1.75.
        # All remaining probabilities stay EXACTLY D; no renormalization artefact.
        pairs = np.linspace(0,K//2-1,10,dtype=int)
        for pair in pairs:
            j = 2*pair
            delta = .75*D[j]
            q[j] -= delta
            q[j+1] += delta
    assert np.isclose(q.sum(),1) and np.all(q>0)
    return donor,D,q

def simulate(q, bottlenecks, depths, rng):
    """Independent outer simulation of the manuscript founder-growth-read model."""
    X=np.zeros((N_HOSTS,len(q)),dtype=np.int64)
    for m,(B,n) in enumerate(zip(bottlenecks,depths)):
        C=rng.multinomial(int(B),q)
        active=C>0
        Q=rng.dirichlet(C[active].astype(float))
        X[m,active]=rng.multinomial(int(n),Q)
    assert np.array_equal(X.sum(axis=1),depths)
    return X

def cp(k,n):
    """Pointwise exact 95% binomial interval across independent datasets."""
    if n==0: return np.nan,np.nan
    return (0. if k==0 else beta.ppf(.025,k,n-k+1),
            1. if k==n else beta.ppf(.975,k+1,n-k))

def mean_summary(values):
    """Dataset-level MC SE and approximate t interval; never pool variants."""
    a=np.asarray(values,float); n=len(a); mean=a.mean()
    se=a.std(ddof=1)/np.sqrt(n) if n>1 else np.nan
    half=t.ppf(.975,n-1)*se if n>1 else np.nan
    return dict(estimate=mean,mc_se=se,ci95_low=max(0.,mean-half) if n>1 else np.nan,
                ci95_high=min(1.,mean+half) if n>1 else np.nan,
                interval='approximate dataset-level t; unreliable with few/zero events',n_datasets=n)

def summarize(out):
    out=Path(out); rates=[]; feature_rows=[]; metrics=[]; fits=[]; status=[]
    config=json.loads((out/'config.json').read_text())
    for case in config['scenarios']:
        name=case['name']; files=sorted((out/name).glob('rep_*.csv'))
        status.append(dict(scenario=name,completed=len(files),requested=config['n_datasets'],
                           failures=len(list((out/name).glob('rep_*.error.txt')))))
        if not files: continue
        df=pd.concat([pd.read_csv(f).assign(dataset=int(f.stem.split('_')[1])) for f in files],ignore_index=True)
        for alpha in ALPHAS:
            # Each fixed feature has one observation per independent dataset.
            for feature,g in df.groupby('feature'):
                k=int((g.pval<=alpha).sum()); low,high=cp(k,len(g))
                feature_rows.append(dict(scenario=name,feature=feature,D=g.D.iloc[0],q_true=g.q_true.iloc[0],
                    truth=g.truth.iloc[0],alpha=alpha,rejections=k,n_datasets=len(g),rate=k/len(g),
                    ci95_low=low,ci95_high=high))
            null=df[df.truth=='neutral']
            per_dataset=null.assign(hit=null.pval<=alpha).groupby('dataset').hit.mean()
            rates.append(dict(scenario=name,alpha=alpha,**mean_summary(per_dataset)))
            # qval_FDR is the BH-adjusted p-value, reusable at each threshold.
            fdp=[]; power=[]; directional=[]; any_reject=[]
            for _,g in df.groupby('dataset'):
                reject=g.qval_FDR<=alpha; true_alt=g.truth!='neutral'
                V=int((reject & ~true_alt).sum()); R=int(reject.sum())
                fdp.append(V/max(R,1)); any_reject.append(R>0)
                if true_alt.any():
                    power.append(float((reject & true_alt).sum()/true_alt.sum()))
                    right=((g.truth=='enriched') & (g.direction=='up_in_recipient')) | ((g.truth=='depleted') & (g.direction=='down_in_recipient'))
                    directional.append(float((reject & true_alt & right).sum()/true_alt.sum()))
            if not case['alternative']:
                k=sum(any_reject); low,high=cp(k,len(any_reject))
                metrics.append(dict(scenario=name,alpha=alpha,metric='BH FDR = P(any rejection), complete null',
                    estimate=k/len(any_reject),ci95_low=low,ci95_high=high,n_datasets=len(any_reject),interval='exact binomial'))
            else:
                for label,values in [('BH FDR',fdp),('BH power',power),('BH direction-correct power',directional)]:
                    metrics.append(dict(scenario=name,alpha=alpha,metric=label,**mean_summary(values)))
        for f in sorted((out/name).glob('rep_*.npz')):
            with np.load(f) as a:
                for m in range(N_HOSTS):
                    fits.append(dict(scenario=name,dataset=int(f.stem.split('_')[1]),host=m+1,
                                     B_true=a['B_true'][m],B_hat=a['B_hat'][m]))
    for name,rows in [('completion',status),('calibration_mean',rates),('calibration_by_feature',feature_rows),('bh_metrics',metrics),('bottlenecks',fits)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    if rates:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,5))
        for name,g in pd.DataFrame(rates).groupby('scenario'):
            if not name.startswith('null_'): continue
            ax.errorbar(g.alpha,g.estimate,yerr=1.96*g.mc_se,marker='o',capsize=3,label=name)
        ax.plot([0,.1],[0,.1],'k--',label='nominal level')
        ax.set(xlabel='Nominal unadjusted level',ylabel='Mean null rejection rate',ylim=(0,None))
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'null_calibration.png',dpi=180);plt.close(fig)
    return pd.DataFrame(rates),pd.DataFrame(metrics)

def run(n_datasets=200,output='bn_validation_results',seed=12345,n_jobs=2,scenario_names=None):
    if n_datasets<1: raise ValueError('n_datasets must be positive')
    cases=[c for c in SCENARIOS if scenario_names is None or c['name'] in scenario_names]
    if not cases: raise ValueError('No matching scenario')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(Path(bn.__file__).read_bytes()).hexdigest()
    config=dict(n_datasets=int(n_datasets),n_boot=N_BOOT,n_hosts=N_HOSTS,seed=int(seed),
                n_jobs=int(n_jobs),scenarios=cases,module_sha256=digest,
                reestimate_bottleneck=True,pseudocount=0.,alphas=ALPHAS,
                numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version())
    encoded=json.dumps(config,indent=2)
    if (out/'config.json').exists() and (out/'config.json').read_text()!=encoded:
        raise ValueError('Configuration changed: choose a new output folder.')
    (out/'config.json').write_text(encoded)
    failures=0
    for case in cases:
        folder=out/case['name'];folder.mkdir(exist_ok=True)
        donor,D,q=design(case)
        truth=np.where(q<D,'depleted',np.where(q>D,'enriched','neutral'))
        scenario_index=next(i for i,c in enumerate(SCENARIOS) if c['name']==case['name'])
        for r in range(n_datasets):
            stem=folder/f'rep_{r:05d}'
            if stem.with_suffix('.csv').exists(): continue
            children=np.random.SeedSequence([seed,scenario_index,r]).spawn(2)
            X=simulate(q,case['B'],case['depths'],np.random.default_rng(children[0]))
            test_seed=int(children[1].generate_state(1)[0])
            start=time.perf_counter()
            print(f'{case["name"]}: dataset {r+1}/{n_datasets}, 200 bootstraps',flush=True)
            try:
                result,diag=bn.neutrality_bootstrap(donor,X,n_boot=N_BOOT,
                    reestimate_bottleneck=True,seed=test_seed,feature_ids=list(range(case['K'])),
                    pseudocount=0.,n_jobs=n_jobs,progress_every=0,lr_backend='cpp')
                result=result.set_index('feature').reindex(range(case['K'])).reset_index()
                if not result.tested.all() or not np.isfinite(result.pval).all():
                    raise RuntimeError('Some variants untested or nonfinite; do not silently omit.')
                result['D']=D;result['q_true']=q;result['truth']=truth
                result['dataset']=r;result['scenario']=case['name']
                np.savez_compressed(stem.with_suffix('.npz'),donor=donor,X=X,q_true=q,
                    B_true=case['B'],B_hat=diag['observed_bottlenecks'].to_numpy(),
                    B_boot=diag['bootstrap_bottlenecks'].to_numpy(),bootstrap_seed=test_seed)
                temp=stem.with_suffix('.tmp');result.to_csv(temp,index=False);temp.replace(stem.with_suffix('.csv'))
                stem.with_suffix('.error.txt').unlink(missing_ok=True)
                print(f'  completed in {time.perf_counter()-start:.1f} s',flush=True)
            except Exception:
                import traceback
                stem.with_suffix('.error.txt').write_text(traceback.format_exc())
                failures+=1;print('  FAILED: saved traceback; will retry when resumed.',flush=True)
            summarize(out)
    summaries=summarize(out)
    if failures: print('WARNING: failures occurred. Summaries of completed runs alone may be biased; inspect completion.csv.')
    return summaries

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-datasets',type=int,default=200)
    p.add_argument('--output',default='bn_validation_results')
    p.add_argument('--seed',type=int,default=12345)
    p.add_argument('--n-jobs',type=int,default=2)
    p.add_argument('--scenarios',nargs='+',choices=[c['name'] for c in SCENARIOS])
    a=p.parse_args();run(a.n_datasets,a.output,a.seed,a.n_jobs,a.scenarios)
