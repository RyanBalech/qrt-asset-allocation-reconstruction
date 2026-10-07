from overlap_model import load,ROOT
import numpy as np
import pandas as pd

def whiten(R,ids,reg=.1):
    rr=R[:,ids,:].transpose(0,2,1).astype(float)
    ref=rr[:2522].reshape(-1,len(ids))
    ref=ref[np.isfinite(ref).all(axis=1)]
    scale=np.std(ref,axis=0)
    cov=np.cov(ref/scale,rowvar=False)
    eig,U=np.linalg.eigh(cov)
    W=(U/np.sqrt(np.maximum(eig,reg)))@U.T
    return np.nan_to_num(rr/scale)@W

def candidates(A,pool):
    rows=[]
    for h in range(1,20):
        aa=A[pool,:20-h].reshape(len(pool),-1).copy()
        bb=A[pool,h:].reshape(len(pool),-1).copy()
        aa/=np.linalg.norm(aa,axis=1,keepdims=True)+1e-12
        bb/=np.linalg.norm(bb,axis=1,keepdims=True)+1e-12
        sim=aa@bb.T
        np.fill_diagonal(sim,-2)
        js=np.argsort(sim,axis=1)[:,-2:][:,::-1]
        for i,q in enumerate(pool):
            for rank,j in enumerate(js[i]):rows.append((q,pool[j],h,float(sim[i,j]),rank))
    return pd.DataFrame(rows,columns=['q','ref','gap','sim','rank'])

if __name__=='__main__':
    R,Y,dates,allocs,d,meta,order=load()
    output=[]
    for g in range(1,5):
        ids=np.flatnonzero((meta.group.values==g)&(meta.n.values>500)&(meta.sv1.values<.9))
        allids=np.flatnonzero(meta.group.values==g)
        test=np.flatnonzero((dates>'DATE_2522')&np.isfinite(R[:,ids[0],0]))
        for reg in [.03,.1,.3]:
            A=whiten(R,ids,reg)
            c=candidates(A,test)
            c['GROUP']=g;c['reg']=reg;output.append(c)
            best=c.loc[c.groupby('q').sim.idxmax()]
            print('TEST',g,reg,[(s,int((best.sim>s).sum())) for s in [.3,.5,.6,.7,.8,.9]],flush=True)
            for seed in [42,2026,123]:
                old=pd.read_parquet(ROOT/f'validation_candidates_g{g}_{seed}.parquet')
                pool=np.sort(old.q.unique())
                cc=candidates(A,pool)
                cc['GROUP']=g;cc['seed']=seed;cc['reg']=reg
                cc.to_parquet(ROOT/f'validation_white_g{g}_{seed}_{reg}.parquet',index=False)
                best=cc.loc[cc.groupby('q').sim.idxmax()]
                yy=Y[best.q.values][:,allids]
                pp=R[best.ref.values][:,allids][np.arange(len(best)),:,best.gap.values-1]
                for s in [.5,.6,.7,.8]:
                    m=np.isfinite(yy)&np.isfinite(pp)&(best.sim.values[:,None]>s)
                    print('VALID',g,reg,seed,s,'coverage',m.sum()/np.isfinite(yy).sum(),'acc',np.mean((yy[m]>0)==(pp[m]>0)),flush=True)
    pd.concat(output,ignore_index=True).to_parquet(ROOT/'test_candidates_white.parquet',index=False)
