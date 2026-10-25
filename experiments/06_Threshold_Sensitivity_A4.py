"""Threshold sensitivity."""

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
from dataclasses import asdict
df=r.select_mode(r.load_data(),MODE)
tr,te=r.partitions(df,'profile',42)
out=r.result_folder('threshold_sensitivity',MODE)
records=[]
for coefficient in r.COEFFICIENTS:
  labels=r.label_by_coefficient(df,coefficient)
  for name in r.MODEL_NAMES:
    for seed in SEEDS:
      print('Threshold',coefficient,name,seed,flush=True)
      Xtr=df.loc[:,r.FEATURES].iloc[tr];Xte=df.loc[:,r.FEATURES].iloc[te]
      ytr=labels[tr];yte=labels[te]
      params,score,tuning_seconds=r.hyper_tune(name,Xtr,ytr,seed,CFG)
      folder=r.mkdir(out/f'coefficient_{coefficient:g}'/name/f'seed_{seed}')
      init,shap_seconds=r.shap_initial_weights(name,params,Xtr,ytr,seed,CFG,
                    folder/'shap_initialisation.npz')
      optimum=r.optimize_cma(name,params,Xtr,ytr,seed,CFG,init)
      fit=r.fit_test(name,Xtr,ytr,Xte,yte,seed,params,True,optimum['weights'])
      record={'coefficient':coefficient,'model':name,'seed':seed,
              'class0_percent':100*np.mean(labels==0),'changed_percent':100*np.mean(labels!=r.label_by_coefficient(df,1.5)),
              **{k:v for k,v in fit['metrics'].items() if k!='confusion_matrix'},
              'tuning_seconds':tuning_seconds,'shap_seconds':shap_seconds,
              'optimise_seconds':optimum['search_seconds'],
              'n_weight_evals':optimum['n_evaluations']}
      records.append(record)
      r.write_json(folder/'run.json',{'metrics':record,'tuned_params':params,
                    'initial_weights':init,'optimized':optimum})
      pd.DataFrame(records).to_csv(out/'repetition_results.csv',index=False)
summary=r.summarize_trials(out/'repetition_results.csv',['coefficient','model'],['macro_f1','class0_f1'])
summary.to_csv(out/'summary.csv',index=False)
print(summary.to_string(index=False))
