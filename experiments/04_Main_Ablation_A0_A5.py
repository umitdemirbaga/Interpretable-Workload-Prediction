"""Ablation experiments: A0–A5."""

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


import pandas as pd, numpy as np
from pathlib import Path
df=r.select_mode(r.load_data(),MODE)
train,test=r.partitions(df,'job',42)
Xtr=df.loc[:,r.FEATURES].iloc[train];Xte=df.loc[:,r.FEATURES].iloc[test]
ytr=df.performance.iloc[train];yte=df.performance.iloc[test]
out=r.result_folder('ablation',MODE)
results=[]
for name in r.MODEL_NAMES:
  for seed in SEEDS:
    tuned=r.read_tuned(MODE,name,seed)
    seed_folder=r.mkdir(out/name/f'seed_{seed}')
    record_by_config={}
    def run_one(config,params=None,scaled=True,weights=None,extra=None):
        fitted=r.fit_test(name,Xtr,ytr,Xte,yte,seed,params,scaled,weights)
        record=dict(model=name,seed=seed,configuration=config,
                    train_seconds=fitted['train_seconds'],
                    test_predict_seconds=fitted['test_predict_seconds'],
                    **{k:v for k,v in fitted['metrics'].items() if k!='confusion_matrix'})
        if extra:record.update(extra)
        r.save_model(seed_folder/f'{config}.joblib',fitted)
        pd.DataFrame({'test_index':test,'actual':yte.to_numpy(),'predicted':fitted['predictions']}).to_csv(seed_folder/f'{config}_predictions.csv',index=False)
        r.write_json(seed_folder/f'{config}.json',{**record,'confusion_matrix':fitted['metrics']['confusion_matrix'],
              'params':params,'weights':weights})
        results.append(record);record_by_config[config]=record
        pd.DataFrame(results).to_csv(out/'repetition_results.csv',index=False)
        print(name,seed,config,'macro-F1',round(record['macro_f1'],5),flush=True)
    run_one('A0',scaled=False)
    run_one('A1',scaled=True)
    run_one('A2',params=tuned)
    uniform=np.ones(len(r.FEATURES))/len(r.FEATURES)
    start_uniform=r.softmax(np.zeros(len(r.FEATURES)))
    a3=r.optimize_cma(name,tuned,Xtr,ytr,seed,CFG,start_uniform)
    run_one('A3',params=tuned,weights=a3['weights'],extra={'optimise_seconds':a3['search_seconds'],'fitness_calls':a3['n_evaluations']})
    init,shap_seconds=r.shap_initial_weights(name,tuned,Xtr,ytr,seed,CFG,
               archive_path=seed_folder/'shap_initialisation.npz')
    a4=r.optimize_cma(name,tuned,Xtr,ytr,seed,CFG,init)
    run_one('A4',params=tuned,weights=a4['weights'],extra={'shap_seconds':shap_seconds,
               'optimise_seconds':a4['search_seconds'],'fitness_calls':a4['n_evaluations']})
    a5=r.optimize_random(name,tuned,Xtr,ytr,seed,CFG,a4['n_evaluations'])
    run_one('A5',params=tuned,weights=a5['weights'],extra={'optimise_seconds':a5['search_seconds'],'fitness_calls':a5['n_evaluations']})
    r.write_json(seed_folder/'optimisation.json',{'A3':a3,'A4':a4,'A5':a5,
                'SHAP_initial_weights':init})
print('Saved repetition results:',out/'repetition_results.csv')


raw=pd.read_csv(out/'repetition_results.csv')
cols=['accuracy','macro_precision','macro_recall','macro_f1','class0_f1']
summary=r.summarize_trials(out/'repetition_results.csv',['model','configuration'],cols)
summary.to_csv(out/'summary.csv',index=False)
print(summary.to_string(index=False))
