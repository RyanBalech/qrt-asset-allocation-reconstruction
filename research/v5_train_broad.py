from v5_train_meta import *

def selected_features(frame):
    x=frame.copy();k=x.group.eq(3)
    for c in ['external','market_external_mean','external_local_rank']:
        x.loc[k,c]=x.loc[k,'previous_'+c]
    x['external_change']=x.external-x.previous_external
    return x

if __name__=='__main__':
    manifest=json.loads((ROOT/'v5_pool_manifest.json').read_text())
    train=selected_features(pd.concat([pd.read_parquet(ROOT/f'v5_broad_meta_pool_{k}.parquet') for k,v in manifest.items() if v['role']=='train'],ignore_index=True))
    dev=selected_features(pd.concat([pd.read_parquet(ROOT/f'v5_broad_meta_pool_{k}.parquet') for k,v in manifest.items() if v['role']=='development'],ignore_index=True))
    assert not set(train.q)&set(dev.q)
    reserved=set(sum([v['pool'] for v in manifest.values() if v['role']=='validation'],[]));assert not set(train.q)&reserved
    y=(train.target>0).astype(int);yv=(dev.target>0).to_numpy();reports=[];preds={}
    print('BROAD BASE',float(np.mean((old_signal(dev)>0)==yv)),flush=True)
    for mode,depth,iterations in [('simple',4,500),('full',5,650),('full',6,900)]:
        x=prepare(train,mode);v=prepare(dev,mode);name=f'broad_{mode}_d{depth}'
        model=CatBoostClassifier(iterations=iterations,depth=depth,learning_rate=.035,l2_leaf_reg=15,loss_function='Logloss',thread_count=5,random_seed=20260921,verbose=False,allow_writing_files=False)
        model.fit(x,y);model.save_model(str(ROOT/f'v5_meta_{name}.cbm'))
        p=model.predict_proba(v)[:,1];preds[name]=p
        for k,ix in [('all',np.ones(len(dev),bool)),('covered',dev.covered>.5),('uncovered',dev.covered<.5)]:
            rec={'model':name,'subset':k,'accuracy':float(np.mean((p[ix]>.5)==yv[ix]))};reports.append(rec);print(rec,flush=True)
        pd.Series(model.feature_importances_,index=x.columns).sort_values(ascending=False).to_csv(ROOT/f'v5_meta_importance_{name}.csv')
    np.savez_compressed(ROOT/'v5_broad_dev_predictions.npz',**preds,q=dev.q.to_numpy(),y=yv,base=old_signal(dev))
    pd.DataFrame(reports).to_csv(ROOT/'v5_broad_development.csv',index=False)
