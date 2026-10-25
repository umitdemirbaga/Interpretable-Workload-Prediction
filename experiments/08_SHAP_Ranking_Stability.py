"""SHAP ranking stability."""

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
from itertools import combinations
from scipy.stats import spearmanr,kendalltau
base=r.result_folder('threshold_sensitivity',MODE)
rows=[];pairs=[]
for path in sorted(base.glob('coefficient_*/*/seed_*/shap_initialisation.npz')):
    with np.load(path,allow_pickle=False) as data:
        values=np.asarray(data['values']);importance=np.abs(values).mean(axis=0)
    coef=float(path.parents[2].name.replace('coefficient_',''))
    model=path.parents[1].name;seed=int(path.parent.name.replace('seed_',''))
    for feature,v in zip(r.FEATURES,importance):
        rows.append({'coefficient':coef,'model':model,'seed':seed,'feature':feature,'mean_abs_shap':v})
if not rows:raise FileNotFoundError('SHAP attribution arrays not found.')
table=pd.DataFrame(rows)
for (coef,name),sub in table.groupby(['coefficient','model']):
    wide=sub.pivot(index='feature',columns='seed',values='mean_abs_shap').reindex(r.FEATURES)
    for a,b in combinations(wide.columns,2):
        rho=spearmanr(wide[a],wide[b]).statistic
        tau=kendalltau(wide[a],wide[b]).statistic
        pairs.append({'coefficient':coef,'model':name,'seed_a':a,'seed_b':b,
                      'spearman':rho,'kendall':tau})
out=r.result_folder('shap_stability',MODE)
table.to_csv(out/'feature_importances.csv',index=False)
pd.DataFrame(pairs).to_csv(out/'pairwise_rank_agreement.csv',index=False)
print(pd.DataFrame(pairs).groupby(['coefficient','model'])[['spearman','kendall']].agg(['mean','std']))
