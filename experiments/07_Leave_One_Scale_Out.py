"""Leave-one-scale-out evaluation."""

from pathlib import Path
import os
import sys
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from src import repro as r
MODE = os.environ.get("EXPERIMENT_MODE", "full")
CFG = r.settings(MODE)
SEEDS = r.SEEDS if MODE == "full" else (11,)


import numpy as np,pandas as pd
df=r.select_mode(r.load_data(),MODE)
path=PROJECT/'data/metadata/job_scale_map.csv'
if not path.exists():
    raise FileNotFoundError(f'{path} is required. Consult data/metadata/README.md.')
meta=pd.read_csv(path)
assert {'group','scale','original_job_id'}.issubset(meta.columns)
assert meta.group.is_unique and not meta.original_job_id.isna().any()
if MODE=='full':assert len(meta)==800
assert set(df.group).issubset(set(meta.group))
assert set(meta.scale)=={'250K','500K','750K','1M'}
frame=df.merge(meta[['group','scale','original_job_id']],on='group',validate='many_to_one',sort=False)
assert len(frame)==len(df)
print(frame.scale.value_counts())


out=r.result_folder('leave_one_scale_out',MODE)
records=[]
for held_out in ('250K','500K','750K','1M'):
    test=frame.scale.eq(held_out).to_numpy();train=~test
    assert set(frame.original_job_id[train]).isdisjoint(set(frame.original_job_id[test]))
    Xtr=frame.loc[train,r.FEATURES];Xte=frame.loc[test,r.FEATURES]
    ytr=frame.loc[train,'performance'];yte=frame.loc[test,'performance']
    for name in r.MODEL_NAMES:
      for seed in SEEDS:
        base=r.fit_test(name,Xtr,ytr,Xte,yte,seed,scaled=False)
        records.append(dict(held_out_scale=held_out,model=name,seed=seed,configuration='A0',
                         **{k:v for k,v in base['metrics'].items() if k!='confusion_matrix'}))
        params,_,_=r.hyper_tune(name,Xtr,ytr,seed,CFG)
        init,_=r.shap_initial_weights(name,params,Xtr,ytr,seed,CFG)
        optimum=r.optimize_cma(name,params,Xtr,ytr,seed,CFG,init)
        optimized=r.fit_test(name,Xtr,ytr,Xte,yte,seed,params,True,optimum['weights'])
        records.append(dict(held_out_scale=held_out,model=name,seed=seed,configuration='A4',
                         **{k:v for k,v in optimized['metrics'].items() if k!='confusion_matrix'}))
        pd.DataFrame(records).to_csv(out/'repetition_results.csv',index=False)
summary=r.summarize_trials(out/'repetition_results.csv',['held_out_scale','model','configuration'],
                           ['accuracy','macro_precision','macro_recall','macro_f1'])
summary.to_csv(out/'summary.csv',index=False)
print(summary.to_string(index=False))
