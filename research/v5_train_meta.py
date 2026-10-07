"""Fit reconstruction correction on disjoint date pools; tune on earlier v4 checks."""
from overlap_model import ROOT
from catboost import CatBoostClassifier
from scipy.special import ndtr,expit
import numpy as np,pandas as pd,json

def old_signal(x):
    return np.where(x.covered>.5,x.direct,x.external+.3*x.baseline)

def prepare(x,mode='full'):
    names=[c for c in x.columns if c not in ['q','target','pool']]
    if mode=='simple':names=[c for c in names if not c.startswith(('ret_','volume_','donor_ret','allocation'))]
    return x[names]

if __name__=='__main__':
    manifest=json.loads((ROOT/'v5_pool_manifest.json').read_text())
    train=pd.concat([pd.read_parquet(ROOT/f'v5_meta_pool_{k}.parquet') for k,v in manifest.items() if v['role']=='train'],ignore_index=True)
    dev=pd.concat([pd.read_parquet(ROOT/f'v5_meta_pool_{k}.parquet') for k,v in manifest.items() if v['role']=='development'],ignore_index=True)
    assert not set(train.q)&set(dev.q)
    reserved=set(sum([v['pool'] for v in manifest.values() if v['role']=='validation'],[]))
    assert not set(train.q)&reserved and not set(dev.q)&reserved
    y=(train.target>0).astype(int);yv=(dev.target>0).to_numpy()
    print('TRAIN',train.shape,'DEVELOPMENT',dev.shape,'baseline',np.mean((old_signal(dev)>0)==yv),flush=True)
    reports=[];preds={}
    for mode,depth,iterations in [('simple',4,500),('full',5,650),('full',6,900)]:
        x=prepare(train,mode);v=prepare(dev,mode);name=f'{mode}_d{depth}'
        model=CatBoostClassifier(iterations=iterations,depth=depth,learning_rate=.035,l2_leaf_reg=15,loss_function='Logloss',thread_count=5,random_seed=20260921,verbose=False,allow_writing_files=False)
        model.fit(x,y);model.save_model(str(ROOT/f'v5_meta_{name}.cbm'))
        p=model.predict_proba(v)[:,1];preds[name]=p
        for blend in [0.,.25,.5]:
            oldprob=ndtr(old_signal(dev)/np.where(dev.covered>.5,.55,.6))
            pp=(1-blend)*p+blend*oldprob
            rec={'model':name,'v4_blend':blend,'accuracy':float(np.mean((pp>.5)==yv)),'covered':float(np.mean((pp[dev.covered>.5]>.5)==yv[dev.covered>.5])),'uncovered':float(np.mean((pp[dev.covered<.5]>.5)==yv[dev.covered<.5]))}
            reports.append(rec);print(rec,flush=True)
        pd.Series(model.feature_importances_,index=x.columns).sort_values(ascending=False).to_csv(ROOT/f'v5_meta_importance_{name}.csv')
    np.savez_compressed(ROOT/'v5_meta_dev_predictions.npz',**preds,q=dev.q.to_numpy(),pool=dev.pool.to_numpy(),y=yv,old=old_signal(dev))
    pd.DataFrame(reports).to_csv(ROOT/'v5_meta_development.csv',index=False)
