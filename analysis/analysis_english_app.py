import os
HERE=os.path.dirname(os.path.abspath(__file__))
import pandas as pd, numpy as np, json
from scipy import stats
N_LAT='latency'
def analyse(runs, base, versions, HIGH, LOW, N, label):
    M=HIGH+LOW; b=runs.loc[base]; nb=len(b); q=len(M)
    k=stats.t.ppf(1-0.05/q,nb-1)*np.sqrt(1+1/nb)
    mu,sd=b.mean(),b.std(ddof=1)
    sp={m:(max(sd[m],1/N) if m!=N_LAT else sd[m]) for m in M}
    lim=lambda m,kk:(mu[m]-kk*sp[m]) if m in HIGH else (mu[m]+kk*sp[m])
    def fails(r,kk): return [m for m in M if not pd.isna(r[m]) and ((m in HIGH and r[m]<lim(m,kk)) or (m in LOW and r[m]>lim(m,kk)))]
    out={'k':k,'n_base':nb,'q':q,'bands':{m:dict(mean=mu[m],sd=sd[m],min=b[m].min(),max=b[m].max(),lk=lim(m,k),l2=lim(m,2)) for m in M},'versions':{}}
    print(f'\n=== {label}: k={k:.3f} (q={q}, n={nb})')
    print(pd.DataFrame(out['bands']).T.round(3).to_string())
    # LOO
    loo=0
    for r in b.index:
        bb=b.drop(r); m2,s2=bb.mean(),bb.std(ddof=1); kk=stats.t.ppf(1-0.05/q,nb-2)*np.sqrt(1+1/(nb-1))
        f=[m for m in M if ((m in HIGH and b.loc[r,m]< m2[m]-kk*(max(s2[m],1/N) if m!=N_LAT else s2[m])) or (m in LOW and b.loc[r,m]> m2[m]+kk*(max(s2[m],1/N) if m!=N_LAT else s2[m])))]
        loo+=bool(f)
    out['loo']=loo; print('LOO fails',loo,'of',nb)
    for v in versions:
        x=runs.loc[v]
        fk=[fails(r,k) for _,r in x.iterrows()]; f2=[fails(r,2) for _,r in x.iterrows()]
        fixed=[[m for m in HIGH if m!=N_LAT and not pd.isna(r[m]) and r[m]<0.5] for _,r in x.iterrows()]
        P=[m for m in M if m!=N_LAT and not x[m].isna().all()]
        sr=float(np.mean([any((m in HIGH and r[m]<br[m]) or (m in LOW and r[m]>br[m]) for m in P) for _,r in x.iterrows() for _,br in b.iterrows()]))
        ps=[]
        for m in [m for m in M if m!=N_LAT]:
            if x[m].isna().all(): continue
            p=stats.mannwhitneyu(x[m],b[m],alternative='less' if m in HIGH else 'greater').pvalue; ps.append([m,p,float(x[m].mean()-b[m].mean())])
        ps.sort(key=lambda t:t[1]); n=len(ps); run_=0; eff=[]
        for i,(m,p,dl) in enumerate(ps):
            run_=max(run_,min(1,p*(n-i))); eff.append([m,p,run_,dl])
        sig=[(e[0],round(e[3],3)) for e in eff if e[2]<0.05]
        out['versions'][v]=dict(means=x.mean().to_dict(),k=fk,two=f2,fixed=fixed,single=sr,sig=sig)
        print(f"{v}: k-fail {sum(map(bool,fk))}/5 {sorted(set(sum(fk,[])))} | 2sd {sum(map(bool,f2))}/5 {sorted(set(sum(f2,[])))} | fixed {sum(map(bool,fixed))}/5 | single {sr:.2f} | sig {sig}")
    return out
R={}
# English
e=pd.read_csv(os.path.join(HERE,'..','data','results','english_results.csv'),encoding='utf-8-sig'); print('en',e.shape, e.groupby('version').run.nunique().to_dict())
re_=e.groupby(['version','run'],sort=False)[['retrieval_hit','contains_ref','judge_correct','faithful','lang_consistent','abstained','latency']].mean()
print(re_.groupby(level=0,sort=False).mean().round(3).to_string())
R['en']=analyse(re_,'en_baseline',['en_control','en_K2','en_R1'],['retrieval_hit','contains_ref','judge_correct','lang_consistent'],['abstained','latency'],50,'English')
# App
a=pd.read_csv(os.path.join(HERE,'..','data','results','llamaindex_app_results.csv'),encoding='utf-8-sig')
ra=a.groupby(['version','run'],sort=False)[['retrieval_hit','contains_ref','judge_correct','lang_consistent','abstained','latency']].mean()
R['app']=analyse(ra,'app_baseline',['app_control','app_K2'],['retrieval_hit','contains_ref','judge_correct','lang_consistent'],['abstained','latency'],100,'App')
json.dump(R,open(os.path.join(HERE,'..','results','stats_english_app.json'),'w'),indent=1,default=lambda o: float(o) if hasattr(o,'__float__') else str(o))
