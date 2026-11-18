import os
HERE=os.path.dirname(os.path.abspath(__file__))
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e'})
B1,O2,GR,INK='#2a78d6','#eb6834','#8a8984','#0b0b0b'
d=pd.read_csv(os.path.join(HERE,'..','data','results','arabic_results.csv'),encoding='utf-8-sig')
HIGH=['retrieval_hit','contains_ref','judge_correct','arabic_only']
M=HIGH+['abstained','latency']
runs=d.groupby(['version','run'],sort=False)[M].mean(); b=runs.loc['baseline']; c=runs.loc['control']
mu,sd=b.mean(),b.std(ddof=1); k=stats.t.ppf(1-0.05/6,9)*np.sqrt(1.1)
lim=lambda m,kk: mu[m]-kk*max(sd[m],0.01)
# Figure 4.1
names={'contains_ref':'Reference containment','judge_correct':'Judged correctness','arabic_only':'Language consistency'}
fig,ax=plt.subplots(1,3,figsize=(9,3.4))
rng=np.random.default_rng(1)
for a,m in zip(ax,names):
    a.scatter(np.zeros(10)+rng.uniform(-.08,.08,10),b[m],s=36,color=B1,label='Baseline runs',zorder=3,edgecolor='white',linewidth=1)
    a.scatter(np.ones(5)+rng.uniform(-.08,.08,5),c[m],s=36,color=O2,label='No-change control runs',zorder=3,edgecolor='white',linewidth=1)
    a.axhline(lim(m,2),color=GR,ls='--',lw=1.2,label='Limit, two standard deviations')
    a.axhline(lim(m,k),color=INK,ls='-',lw=1.2,label=f'Limit, k = {k:.2f}')
    a.set_xticks([0,1]); a.set_xticklabels(['Baseline','Control']); a.set_xlim(-.6,1.6)
    a.set_title(names[m],fontsize=10,color=INK); a.grid(axis='y',color='#e6e5e0',lw=.6); a.set_axisbelow(True)
ax[0].set_ylabel('Run score (mean over 100 questions)')
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=4,frameon=False,fontsize=9)
fig.tight_layout(rect=(0,.1,1,1)); fig.savefig(os.path.join(HERE,'..','results','figures','figure4_1.png'),dpi=200); plt.close()
# Figure 4.2
order=['baseline','control','P1','M1','P2','M2','K1','K2','R1','R2','no_retrieval']
lab=['Baseline','Control','T1','M1','T2','M2','K1','K2','R1','R2','No\nretrieval']
L={m:(mu[m]-k*max(sd[m],0.01) if m in HIGH else mu[m]+k*max(sd[m],0.01)) for m in M}
def failed(r):
    return any((m in HIGH and not pd.isna(r[m]) and r[m]<L[m]) or (m not in HIGH and r[m]>L[m]) for m in M)
fig,a=plt.subplots(figsize=(9,3.8))
for i,v in enumerate(order):
    x=runs.loc[v]
    for j,(_,r) in enumerate(x.iterrows()):
        col=O2 if (v!='baseline' and failed(r)) else B1
        a.scatter(i+rng.uniform(-.12,.12),r['judge_correct'],s=34,color=col,edgecolor='white',linewidth=1,zorder=3)
a.axhline(L['judge_correct'],color=INK,lw=1.2); a.text(1.5,L['judge_correct']-.045,'Band limit for judged correctness',ha='center',fontsize=8.5,color=INK)
a.set_xticks(range(len(order))); a.set_xticklabels(lab); a.set_ylabel('Judged correctness of the run'); a.set_ylim(-.03,.8)
a.grid(axis='y',color='#e6e5e0',lw=.6); a.set_axisbelow(True)
from matplotlib.lines import Line2D
a.legend(handles=[Line2D([],[],marker='o',ls='',color=B1,label='Run passed'),Line2D([],[],marker='o',ls='',color=O2,label='Run failed (any metric outside its band)')],frameon=False,loc='lower left',fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE,'..','results','figures','figure4_2.png'),dpi=200); plt.close()
print('k',k)
