"""SHAP and LIME explanations."""

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


import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
df=r.select_mode(r.load_data(),MODE)
tr,te=r.partitions(df,'job',42)
Xtr=df.loc[:,r.FEATURES].iloc[tr].to_numpy()
Xte=df.loc[:,r.FEATURES].iloc[te].to_numpy()
seed=SEEDS[0];out=r.result_folder('explanations',MODE)
figure_dir=r.mkdir(PROJECT/'figures')
for name in r.MODEL_NAMES:
    model_path=r.result_folder('ablation',MODE)/name/f'seed_{seed}'/'A4.joblib'
    if not model_path.exists():raise FileNotFoundError(f'Missing model file: {model_path}')
    bundle=joblib.load(model_path); model=bundle['model']; scaler=bundle['scaler']
    mult=np.asarray(bundle['weights'])*len(r.FEATURES)
    transform=lambda z:scaler.transform(z)*mult
    bg=transform(Xtr[:min(50,len(Xtr))]);ex=transform(Xte[:min(60,len(Xte))])
    if name=='XGBoost':
        explainer=shap.TreeExplainer(model,data=bg,model_output='probability',feature_perturbation='interventional')
        sv=-np.asarray(explainer.shap_values(ex))
    else:
        explainer=shap.KernelExplainer(lambda z:-model.decision_function(z),bg)
        sv=np.asarray(explainer.shap_values(ex,nsamples=CFG.kernel_shap_nsamples,silent=True))
    importance=np.abs(sv).mean(axis=0)
    pd.DataFrame({'feature':r.FEATURES,'mean_abs_shap_class0':importance}).to_csv(out/f'{name}_SHAP_importance.csv',index=False)
    np.savez_compressed(out/f'{name}_SHAP.npz',values=sv,features=np.array(r.FEATURES))
    order=np.argsort(importance)
    fig,ax=plt.subplots(figsize=(8,4));ax.barh(np.array(r.FEATURES)[order],importance[order]);ax.set_xlabel('Mean |SHAP| for class 0');fig.tight_layout()
    fig.savefig(figure_dir/f'{MODE}_{name}_shap.pdf',dpi=600,bbox_inches='tight');plt.close(fig)
    # Explain one original feature vector; LIME receives probability predictions.
    predict_proba=(lambda z:model.predict_proba(transform(z))) if hasattr(model,'predict_proba') else None
    if predict_proba is not None:
        lime=LimeTabularExplainer(Xtr,feature_names=list(r.FEATURES),class_names=['degraded','normal'],
                                  mode='classification',random_state=42)
        explanation=lime.explain_instance(Xte[0],predict_proba,labels=(0,),num_features=8,num_samples=1000)
        r.write_json(out/f'{name}_lime_local.json',{'target_class':0,'weights':explanation.as_list(label=0)})
    else:
        print('SVM RBF decision function is not probability-calibrated; LIME local probability explanation skipped.')
