import pandas as pd, numpy as np, json
from scipy import stats
import os
HERE=os.path.dirname(os.path.abspath(__file__))
d=pd.read_csv(os.path.join(HERE,'..','data','results','arabic_results.csv'),encoding='utf-8-sig')
HIGH=['retrieval_hit','contains_ref','judge_correct','arabic_only']   # faithful excluded: failed human validation (kappa < 0.61)
M=HIGH+['abstained','latency']; N=100
ALL=['retrieval_hit','contains_ref','judge_correct','faithful','arabic_only','abstained','latency']
runs=d.groupby(['version','run'],sort=False)[ALL].mean(); b=runs.loc['baseline']
mu,sd=b.mean(),b.std(ddof=1); q=len(M); nb=len(b)
k=float(stats.t.ppf(1-0.05/q,nb-1)*np.sqrt(1+1/nb))
sp={m:(max(sd[m],1/N) if m!='latency' else sd[m]) for m in M}
lim=lambda m,kk,mu=mu,sp=sp:(mu[m]-kk*sp[m]) if m in HIGH else (mu[m]+kk*sp[m])
out={'k':k,'q':q,'bands':{m:dict(mean=mu[m],sd=sd[m],min=b[m].min(),max=b[m].max(),l2=lim(m,2),lk=lim(m,k)) for m in M}}
out['faithful_desc']=dict(mean=mu['faithful'],sd=sd['faithful'],min=b['faithful'].min(),max=b['faithful'].max())
out['vmeans']=runs.groupby(level=0,sort=False).mean().to_dict(orient='index')
eff={}
for v in ['control','P1','P2','K1','K2','R1','R2','M1','M2','no_retrieval']:
    x=runs.loc[v]; ps=[]
    for m in HIGH+['abstained']:
        if x[m].isna().all(): continue
        p=stats.mannwhitneyu(x[m],b[m],alternative='less' if m in HIGH else 'greater').pvalue; ps.append([m,p,float(x[m].mean()-b[m].mean())])
    ps.sort(key=lambda t:t[1]); n=len(ps); run_=0; res=[]
    for i,(m,p,dl) in enumerate(ps):
        run_=max(run_,min(1,p*(n-i))); res.append([m,p,run_,dl])
    eff[v]=res
    if not x['faithful'].isna().all():
        p=stats.mannwhitneyu(x['faithful'],b['faithful'],alternative='less').pvalue
        out.setdefault('faith_test',{})[v]=[float(p),float(x['faithful'].mean()-b['faithful'].mean())]
out['effects']=eff
def failed(r,kk,ms=M):
    return [m for m in ms if not pd.isna(r[m]) and ((m in HIGH and r[m]<lim(m,kk)) or (m not in HIGH and r[m]>lim(m,kk)))]
V={}
for v in ['control','P1','P2','K1','K2','R1','R2','M1','M2','no_retrieval']:
    x=runs.loc[v]
    V[v]=dict(k=[failed(r,k) for _,r in x.iterrows()], two=[failed(r,2) for _,r in x.iterrows()],
      fixed=[[m for m in HIGH if not pd.isna(r[m]) and r[m]<0.5] for _,r in x.iterrows()])
    P=[m for m in HIGH+['abstained'] if not x[m].isna().all()]
    V[v]['single']=float(np.mean([any((m in HIGH and r[m]<br[m]) or (m not in HIGH and r[m]>br[m]) for m in P) for _,r in x.iterrows() for _,br in b.iterrows()]))
out['verdicts']=V
# leave-one-out
loo={'k':0,'two':0,'k_detail':[],'two_detail':[]}
for i in b.index:
    bb=b.drop(i); m_,s_=bb.mean(),bb.std(ddof=1); kk=float(stats.t.ppf(1-0.05/q,nb-2)*np.sqrt(1+1/(nb-1)))
    sp_={m:(max(s_[m],1/N) if m!='latency' else s_[m]) for m in M}
    r=b.loc[i]
    for name,kv in (('k',kk),('two',2)):
        f=[m for m in M if ((m in HIGH and r[m]<m_[m]-kv*sp_[m]) or (m not in HIGH and r[m]>m_[m]+kv*sp_[m]))]
        if f: loo[name]+=1; loo[name+'_detail'].append([int(i),f])
out['loo']=loo
q_=d[d.version=='baseline'].pivot_table(index='qid',columns='run',values='judge_correct').sum(axis=1)
qr=d[d.version=='baseline'].pivot_table(index='qid',columns='run',values='retrieval_hit').mean(axis=1)
out['qstab']=dict(always=int((q_==10).sum()),never=int((q_==0).sum()),unstable=int(((q_>0)&(q_<10)).sum()),never_retr_wrong=int(((q_==0)&(qr==0)).sum()),retr_wrong=int(((q_==0)&(qr==1)).sum()))
a=d[d.version=='baseline'].pivot_table(index='qid',columns='run',values='answer',aggfunc='first')
out['distinct_mean']=float(a.nunique(axis=1).mean()); out['one_answer']=int((a.nunique(axis=1)==1).sum())
json.dump(out,open(os.path.join(HERE,'..','results','stats_arabic.json'),'w'),indent=1,default=float)
print('k',k,'loo',loo)
for v in V: print(v,{kk:(sum(map(bool,vv)) if isinstance(vv,list) else round(vv,3)) for kk,vv in V[v].items()}, sorted({m for f in V[v]['k'] for m in f}), '| two:',[f for f in V[v]['two'] if f])
print({v:[(e[0],round(e[2],4),round(e[3],3)) for e in eff[v] if e[2]<0.05] for v in eff})
print('faith_test',out['faith_test'])
