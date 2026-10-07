"""Reusable v5 inference: feature-only dates, selected public equities, CatBoost."""
from v5_meta_features import features
from v5_broad_meta_features import add_prior_features
from v5_train_broad import selected_features
from v5_broad_probe import broad_matrices
from v4_equity_imputation import *
from v4_joint_dates import assign
from catboost import CatBoostClassifier
from scipy.special import ndtr
from v5_adaptive_ridge import adaptive
import json
import hashlib

def context():
    R,Y,dates,allocs,d,meta,order=load();panel=np.load(ROOT/'panel.npz')
    z=np.load(ROOT/'baseline_predictions.npz');base=np.zeros_like(Y)
    base[panel['di'],panel['ai']]=np.r_[z['oof'],z['test']]-.5
    turn=np.full_like(Y,np.nan);turn[panel['di'],panel['ai']]=d.MEDIAN_DAILY_TURNOVER.to_numpy()
    cals=sessions();oldxs=stock_matrices(cals,expanded=True);wide=broad_matrices(cals,groups=[1,2,4]);wide[3]=oldxs[3]
    return dict(R=R,Y=Y,dates=dates,allocs=allocs,data=d.drop(columns=['target']),meta=meta,panel=panel,V=panel['V'],base=base,turn=turn,cals=cals,oldxs=oldxs,wide=wide,clockxs=stock_matrices(cals))

def build(ctx,pool,grid,name,adaptive_weights=False):
    pool=np.asarray(pool);grid=pd.DatetimeIndex(grid)
    assignment,_,_=assign(ctx['R'],ctx['V'],ctx['meta'],pool,grid,ctx['cals'],ctx['clockxs'])
    real={int(q):grid[k] for q,k in zip(pool,assignment)}
    cal=pd.DataFrame({'q':pool,'TS':ctx['dates'][pool],'real_date':[str(real[q].date()) for q in pool],'grid_position':assignment})
    cal.to_csv(ROOT/f'v5_calendar_{name}.csv',index=False)
    old,direct,cover=infer(ctx['R'],pool,ctx['meta'],real,ctx['cals'],ctx['oldxs'],alpha=.7,radius=260,tau=100)
    if adaptive_weights:
        ext=adaptive(ctx['R'],pool,ctx['meta'],real,ctx['cals'],ctx['wide'],alpha=.7,radius=260,tau=100,steps=2,power=.5)
    else:
        ext,_,_=infer(ctx['R'],pool,ctx['meta'],real,ctx['cals'],ctx['wide'],alpha=.7,radius=260,tau=100)
    np.savez_compressed(ROOT/f'v5_inference_{name}.npz',pool=pool,external=ext,previous_external=old,direct=direct,coverage=cover)
    prev=features(ctx['R'],ctx['V'],pool,ctx['meta'],real,ctx['cals'],old,direct,cover,ctx['base'],ctx['turn'])
    new=features(ctx['R'],ctx['V'],pool,ctx['meta'],real,ctx['cals'],ext,direct,cover,ctx['base'],ctx['turn'])
    new=add_prior_features(new,prev)
    new['q']=np.repeat(pool,len(ctx['meta']));new['allocation']=np.tile(np.arange(len(ctx['meta'])),len(pool));new['pool']=name
    return new,real,prev

def predict(frame,config,final=False):
    result=[]
    for name in config['models']:
        model=CatBoostClassifier();prefix='v5_final_' if final else 'v5_meta_'
        model.load_model(str(ROOT/f'{prefix}{name}.cbm'))
        result.append(model.predict_proba(frame[model.feature_names_])[:,1])
    p=np.mean(result,axis=0)
    blend=config.get('base_blend',0.)
    if blend:
        signal=np.where(frame.covered>.5,frame.direct,frame.external+.3*frame.baseline)
        p=(1-blend)*p+blend*ndtr(signal/np.where(frame.covered>.5,.55,.6))
    return p

def cached_build(ctx,pool,grid,name,adaptive_weights=False):
    path=ROOT/f'v5_raw_features_{name}.parquet'
    stamp=ROOT/f'v5_raw_features_{name}.json'
    signature=hashlib.sha256((ROOT/'v5_frozen_config.json').read_bytes()).hexdigest()
    if path.exists() and stamp.exists() and json.loads(stamp.read_text())['config_sha256']==signature:
        x=pd.read_parquet(path);cal=pd.read_csv(ROOT/f'v5_calendar_{name}.csv')
        assert np.array_equal(np.asarray(pool),cal.q.to_numpy())
        real={int(row.q):pd.Timestamp(row.real_date) for row in cal.itertuples()}
        prior=x.copy();prior['external']=prior.previous_external
        return x,real,prior
    x,real,prior=build(ctx,pool,grid,name,adaptive_weights)
    x.to_parquet(path,index=False);stamp.write_text(json.dumps({'config_sha256':signature,'adaptive_weights':adaptive_weights}))
    return x,real,prior

def validate():
    cfg=json.loads((ROOT/'v5_frozen_config.json').read_text());ctx=context();truth=mapping()
    manifest=json.loads((ROOT/'v5_pool_manifest.json').read_text());reports=[];dayrows=[]
    for name,entry in manifest.items():
        if entry['role']!='validation':continue
        pool=np.asarray(entry['pool']);x,real,previous=cached_build(ctx,pool,entry['grid'],'holdout_'+name,adaptive_weights=cfg.get('adaptive_weights',False))
        target=ctx['Y'][pool].reshape(-1);keep=np.isfinite(target)
        x=x.loc[keep].copy();x['target']=target[keep];x.to_parquet(ROOT/f'v5_holdout_features_{name}.parquet',index=False)
        p=predict(x,cfg);old=np.where(previous.covered>.5,previous.direct,previous.external+.3*previous.baseline)[keep]
        y=target[keep]>0;correct=(p>.5)==y;oldcorrect=(old>0)==y
        dateaccuracy=np.mean([real[q]==truth[q] for q in pool])
        for g in [0,1,2,3,4]:
            sel=np.ones(len(x),bool) if not g else x.group.eq(g).to_numpy()
            rec={'pool':name,'group':g,'rows':int(sel.sum()),'v5':float(correct[sel].mean()),'v4':float(oldcorrect[sel].mean()),'date_accuracy':float(dateaccuracy),'direct_coverage':float(x.covered.to_numpy()[sel].mean())}
            reports.append(rec)
            if not g:print('HOLDOUT',rec,flush=True)
        a=pd.DataFrame({'q':x.q,'n':1,'new_correct':correct.astype(int),'old_correct':oldcorrect.astype(int)}).groupby('q').sum().reset_index();a['pool']=name;dayrows.append(a)
        np.savez_compressed(ROOT/f'v5_holdout_predictions_{name}.npz',q=x.q.to_numpy(),allocation=x.allocation.to_numpy(),probability=p,old=old,y=y)
        pd.DataFrame(reports).to_csv(ROOT/'v5_holdout_metrics.csv',index=False)
    byday=pd.concat(dayrows).groupby('q')[['n','new_correct','old_correct']].sum();byday.to_csv(ROOT/'v5_holdout_by_date.csv')
    rng=np.random.default_rng(20260922);bootstrap=[]
    for _ in range(3000):
        s=byday.iloc[rng.integers(0,len(byday),len(byday))];bootstrap.append([(s.new_correct/s.n).mul(s.n).sum()/s.n.sum(),(s.new_correct-s.old_correct).sum()/s.n.sum()])
    boot=np.asarray(bootstrap)
    summary={'rows':int(byday.n.sum()),'dates':len(byday),'v5_accuracy':float(byday.new_correct.sum()/byday.n.sum()),'v4_accuracy':float(byday.old_correct.sum()/byday.n.sum()),'accuracy_date_bootstrap_95':np.quantile(boot[:,0],[.025,.975]).tolist(),'improvement_date_bootstrap_95':np.quantile(boot[:,1],[.025,.975]).tolist(),'public_score':None,'note':'Offsets 0,7,14 excluded from correction-model fitting and current-turn development selection. Historical reconstruction using public target-day stock returns; not prospective forecasting. Earlier v4 development and inherited CatBoost OOF priors mean this is not a completely nested, historically untouched evaluation.'}
    (ROOT/'v5_holdout_summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',summary,flush=True)

if __name__=='__main__':validate()
