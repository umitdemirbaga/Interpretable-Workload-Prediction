"""Partial dependence."""

import matplotlib
matplotlib.use("Agg")

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


import numpy as np,pandas as pd,joblib
import matplotlib.pyplot as plt
df=r.select_mode(r.load_data(),MODE)
tr,te=r.partitions(df,'job',42)
reference=df.loc[:,r.FEATURES].iloc[te].to_numpy()[:min(200,len(te))].copy()
out=r.result_folder('pdp',MODE);figure_dir=r.mkdir(PROJECT/'figures');seed=SEEDS[0]
for name in r.MODEL_NAMES:
    config='A0' if name=='XGBoost' else 'A4'
    path=r.result_folder('ablation',MODE)/name/f'seed_{seed}'/f'{config}.joblib'
    if not path.exists():raise FileNotFoundError(f'Missing model file: {path}')
    b=joblib.load(path); model=b['model'];scale=b['scaler']
    mult=np.ones(len(r.FEATURES)) if b['weights'] is None else len(r.FEATURES)*np.asarray(b['weights'])
    rows=[]
    fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for j,(ax,feat) in enumerate(zip(axes.flat,r.FEATURES)):
        grid=np.linspace(np.quantile(reference[:,j],.05),np.quantile(reference[:,j],.95),20)
        points=[]
        for value in grid:
            modified=reference.copy();modified[:,j]=value
            weighted=(modified if scale is None else scale.transform(modified)*mult)
            scores=(model.predict_proba(weighted)[:,0] if name=='XGBoost'
                    else model.decision_function(weighted))
            points.append(float(np.mean(scores)))
            rows.append({'model':name,'feature':feat,'x':value,'mean_score':points[-1]})
        ax.plot(grid,points);ax.set_title(feat);ax.set_xlabel('Original feature value')
        ax.set_ylabel('Mean class-0 probability' if name=='XGBoost' else 'Mean SVM decision score (positive: class 1)')
    fig.savefig(figure_dir/f'{MODE}_{name}_{config}_PDP.pdf',dpi=600,bbox_inches='tight')
    plt.close(fig);pd.DataFrame(rows).to_csv(out/f'{name}_PDP.csv',index=False)
