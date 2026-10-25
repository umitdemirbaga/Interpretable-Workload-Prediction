"""Data integrity."""

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


from importlib.metadata import version
import pandas as pd, numpy as np
path = PROJECT / 'data/processed/All_with_group.csv'
df=r.load_data(path)
manifest=r.environment_manifest(CFG)
manifest['dataset_sha256']=r.file_sha256(path)
manifest['rows']=len(df)
manifest['groups']=df.group.nunique()
manifest['class_counts']={str(k):int(v) for k,v in df.performance.value_counts().sort_index().items()}
r.write_json(PROJECT/'results/dataset_manifest.json',manifest)
print('Dataset:',df.shape,'| Jobs:',df.group.nunique())
print('Classes:',df.performance.value_counts().sort_index().to_dict())
print('Columns:',df.columns.tolist())
print('Makespan sum matches:',np.allclose(df.makespan,df[list(r.TIMES)].sum(axis=1)))


rows=[]
for coefficient in r.COEFFICIENTS:
    labels=r.label_by_coefficient(df,coefficient)
    rows.append({'coefficient':coefficient,'class0_n':int((labels==0).sum()),
                 'class0_percent':round(100*np.mean(labels==0),4),
                 'changed_vs_1_5_percent':round(100*np.mean(labels!=df.performance.to_numpy()),4)})
audit=pd.DataFrame(rows)
audit.to_csv(PROJECT/'results/label_threshold_audit.csv',index=False)
print(audit.to_string(index=False))
assert np.array_equal(r.label_by_coefficient(df,1.5),df.performance.to_numpy())


recomputed_m=df[list(r.TIMES)].sum(axis=1)
low=recomputed_m.groupby(df.group).transform('min')
high=recomputed_m.groupby(df.group).transform('max')
z=(recomputed_m-low)/(high-low)
recomputed=(z<1.5*z.groupby(df.group).transform('median')).astype(int)
print('Differences versus stored labels:',int((recomputed.to_numpy()!=df.performance.to_numpy()).sum()))
print('Use the supplied full-precision normalized_makespan column for exact reproduction.')
