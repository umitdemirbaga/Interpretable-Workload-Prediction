"""Evaluation partitions."""

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


import numpy as np, pandas as pd
df=r.select_mode(r.load_data(),MODE)
for protocol in ('job','profile'):
    train,test=r.store_split(df,protocol=protocol,seed=42) if MODE=='full' else r.partitions(df,protocol=protocol,seed=42)
    train_groups=set(df.group.iloc[train]);test_groups=set(df.group.iloc[test])
    print(protocol,{'train_profiles':len(train),'test_profiles':len(test),
                    'train_groups':len(train_groups),'test_groups':len(test_groups),
                    'overlap_groups':len(train_groups&test_groups)})
    if protocol=='job':assert train_groups.isdisjoint(test_groups)
    if MODE=='full': assert (len(train),len(test))==(19200,4800)


train,test=r.partitions(df,protocol='job',seed=42)
out=r.result_folder('split_audit',MODE)
pd.DataFrame({'original_row_position':train,'group':df.group.iloc[train].to_numpy()}).to_csv(out/'job_train.csv',index=False)
pd.DataFrame({'original_row_position':test,'group':df.group.iloc[test].to_numpy()}).to_csv(out/'job_test.csv',index=False)
print('Saved:',out)
