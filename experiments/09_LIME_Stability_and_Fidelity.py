"""LIME stability and fidelity."""

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
from lime.lime_tabular import LimeTabularExplainer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
from xgboost import XGBClassifier
df=r.select_mode(r.load_data(),MODE)
tr,te=r.partitions(df,'job',42)
Xtr=df.loc[tr,r.FEATURES].to_numpy(); ytr=df.performance.iloc[tr].to_numpy()
train_groups=df.group.iloc[tr].to_numpy()
Xte=df.loc[te,r.FEATURES].to_numpy();instance=Xte[0]
TARGET_CLASS=1
LIME_SEEDS=range(100,130) if MODE=='full' else range(100,103)
NUM_SAMPLES=5000 if MODE=='full' else 250
out=r.result_folder('lime_stability',MODE)
rows=[]
for name in r.MODEL_NAMES:
    if name=='XGBoost': model=r.classifier('XGBoost',42).fit(Xtr,ytr)
    else:
        model=CalibratedClassifierCV(SVC(kernel='linear',C=1.0,class_weight=None),
                                     method='sigmoid',cv=list(GroupKFold(n_splits=3 if MODE=='smoke' else 5).split(Xtr,ytr,groups=train_groups)))
        model.fit(Xtr,ytr)
    # sklearn sorted class order: [0,1]
    target_index=list(model.classes_).index(TARGET_CLASS)
    vectors=[];fidelities=[];local_errors=[]
    original_probability=float(model.predict_proba(instance.reshape(1,-1))[0,target_index])
    for seed in LIME_SEEDS:
        explainer=LimeTabularExplainer(Xtr,feature_names=list(r.FEATURES),class_names=['degraded','normal'],
                                      mode='classification',random_state=seed,discretize_continuous=True,
                                      discretizer='quartile',kernel_width=.75*np.sqrt(len(r.FEATURES)))
        e=explainer.explain_instance(instance,model.predict_proba,labels=(target_index,),
                                     num_features=8,num_samples=NUM_SAMPLES)
        weights=dict(e.local_exp[target_index])
        vector=np.array([weights.get(i,0) for i in range(len(r.FEATURES))])
        vectors.append(vector);fidelities.append(float(e.score))
        approximation=float(np.asarray(e.local_pred).reshape(-1)[0])
        local_errors.append(abs(approximation-original_probability))
        for idx,w in enumerate(vector):rows.append({'model':name,'seed':seed,'feature':r.FEATURES[idx],'weight':w,'lime_r2':e.score})
    vectors=np.asarray(vectors)
    n=len(vectors)
    sign=np.sign(np.where(np.abs(vectors)<=1e-8,0,vectors))
    ranks=pd.DataFrame(np.abs(vectors)).rank(axis=1,ascending=False,method='average').to_numpy()
    top_sets=[]
    for vector in vectors:
        absweights=np.abs(vector)
        cutoff=np.sort(absweights)[-3]
        top_sets.append(set(np.where(absweights>=cutoff)[0]))
    comparisons=[]
    for i,j in combinations(range(n),2):
        corr=(float(np.corrcoef(ranks[i],ranks[j])[0,1])
              if np.std(ranks[i])>0 and np.std(ranks[j])>0 else np.nan)
        comparisons.append({'model':name,'seed_a':list(LIME_SEEDS)[i],
                            'seed_b':list(LIME_SEEDS)[j],
                            'spearman_rho':corr,
                            'top3_jaccard':len(top_sets[i]&top_sets[j])/len(top_sets[i]|top_sets[j]),
                            'sign_agreement':float(np.mean(sign[i]==sign[j]))})
    pairwise=pd.DataFrame(comparisons)
    pairwise.to_csv(out/f'{name}_pairwise.csv',index=False)
    pd.DataFrame({'model':name,'seed':list(LIME_SEEDS),'weighted_local_r2':fidelities,
                 'absolute_profile_probability_error':local_errors}).to_csv(out/f'{name}_fidelity.csv',index=False)
    print(name, 'mean rank Spearman:',pairwise.spearman_rho.mean(),
          'mean top-3 Jaccard:',pairwise.top3_jaccard.mean(),
          'mean sign agreement:',pairwise.sign_agreement.mean(),
          'mean local R2:',np.mean(fidelities),
          'mean absolute profile error:',np.mean(local_errors))
result=pd.DataFrame(rows);result.to_csv(out/'lime_stability.csv',index=False)
