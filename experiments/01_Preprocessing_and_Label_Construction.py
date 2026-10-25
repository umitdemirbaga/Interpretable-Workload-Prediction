"""Preprocessing and label construction."""

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
df=r.load_data()
X=df.loc[:,r.FEATURES].copy()
y=pd.Series(r.label_by_coefficient(df,1.5),name='performance')
assert len(X)==24000 and df.group.nunique()==800
assert np.array_equal(y.to_numpy(),df.performance.to_numpy())
print('Input shape:',X.shape,'| Target:',y.value_counts().sort_index().to_dict())


out=r.result_folder('preprocessing',MODE)
X.to_csv(out/'predictors.csv',index=False)
pd.DataFrame({'group':df.group,'performance':y}).to_csv(out/'group_and_target.csv',index=False)
thresholds=pd.DataFrame({'group':df.group})
for c in r.COEFFICIENTS:
    thresholds[f'performance_{c:g}']=r.label_by_coefficient(df,c)
thresholds.to_csv(out/'threshold_labels.csv',index=False)
print('Saved:',out)
