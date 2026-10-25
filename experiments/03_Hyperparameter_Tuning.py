"""Hyperparameter tuning."""

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
train,test=r.partitions(df,'job',42)
X=df.loc[:,r.FEATURES].iloc[train];y=df.performance.iloc[train]
for name in r.MODEL_NAMES:
    for seed in SEEDS:
        print('Tuning',name,'seed',seed,flush=True)
        params,cv_f1,elapsed=r.hyper_tune(name,X,y,seed,CFG)
        output={'model':name,'seed':seed,'params':params,'mean_cv_class0_f1':cv_f1,
                'search_seconds':elapsed,'split':'job-disjoint','mode':MODE}
        r.write_json(r.tuned_path(MODE,name,seed),output)
        print(output,flush=True)
