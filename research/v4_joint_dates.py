"""Generic joint calendar assignment used for blinded validation and test."""
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix
from v4_stock_clock import evidence
from whiten_matches import whiten
from fused_reconstruction import tensor_scores
import numpy as np
import pandas as pd

def assign(R,V,meta,pool,grid,cals,xs,As=None,threshold=.80,cached=None):
    score=evidence(R,V,meta,cals,xs,pool,grid,alphas=(.7,))[:,:,0] if cached is None else cached[0]
    n=len(pool)
    if As is None:As=[whiten(R,np.flatnonzero((meta.group.values==g)&(meta.n.values>500)&(meta.sv1.values<.9)),.1) for g in range(1,5)]
    tensor=np.stack([tensor_scores(a,pool) for a in As],axis=2) if cached is None else cached[1]
    s=np.nan_to_num(tensor,nan=-2);sim=s.max(3);gap=s.argmax(3)+1
    edges=[]
    for i in range(n):
        for j in range(n):
            g=int(sim[i,j].argmax());val=sim[i,j,g]
            if val>threshold:edges.append((val,i,j,g,int(gap[i,j,g])))
    out={};incoming={}
    for val,i,j,g,h in sorted(edges,reverse=True):
        if i in out or j in incoming:continue
        cursor=j
        while cursor in out:cursor=out[cursor]
        if cursor==i:continue
        out[i]=j;incoming[j]=i
    chains=[]
    for i in range(n):
        if i in incoming:continue
        ch=[i]
        while ch[-1] in out:ch.append(out[ch[-1]])
        chains.append(ch)
    options=[];values=[]
    for ci,ch in enumerate(chains):
        for k in range(n-len(ch)+1):
            ix=k+np.arange(len(ch));values0=score[ch,ix]
            if (values0<-10).any():continue
            good=True
            for off,(i,j) in enumerate(zip(ch[:-1],ch[1:])):
                g=int(sim[i,j].argmax());p=cals[g+1].get_indexer([grid[k+off],grid[k+off+1]])
                if min(p)<0 or p[1]-p[0]!=gap[i,j,g]:good=False;break
            if good:options.append((ci,ix));values.append(values0.sum())
    a=lil_matrix((len(chains)+n,len(options)))
    for col,(ci,ix) in enumerate(options):
        a[ci,col]=1
        for k in ix:a[len(chains)+k,col]=1
    result=milp(-np.asarray(values),integrality=np.ones(len(values)),bounds=Bounds(0,1),constraints=LinearConstraint(a.tocsr(),1,1),options={'time_limit':50})
    if result.x is None:
        if threshold<1.0:
            print('Infeasible calendar edges: increasing feature-only threshold to',round(threshold+.025,3),flush=True)
            return assign(R,V,meta,pool,grid,cals,xs,As,threshold+.025,(score,tensor))
        raise RuntimeError('Joint date assignment failed: '+result.message)
    assignments=np.full(n,-1)
    for v,(ci,ix) in zip(result.x,options):
        if v>.5:assignments[chains[ci]]=ix
    assert len(np.unique(assignments))==n and min(assignments)>=0
    return assignments,score,tensor
