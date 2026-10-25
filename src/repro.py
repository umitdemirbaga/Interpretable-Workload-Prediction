"""Reproducible experiments for duration-defined Hadoop workload-status classification.

Metrics, labels, seeds and candidate distributions follow the submitted manuscript.
The supplied table is processed monitoring data, not raw SmartMonit event logs.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import (GroupShuffleSplit, RandomizedSearchCV,
                                     StratifiedKFold, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier

FEATURES = ('dataSplit', 'map', 'shuffle', 'reduce', 'dataCombine',
            'networkTraffic', 'CPUusage', 'Musage')
TIMES = FEATURES[:5]
SEEDS = (11, 23, 42, 67, 89)
COEFFICIENTS = (1.25, 1.50, 1.75, 2.0)
MODEL_NAMES = ('XGBoost', 'SVM')


def root() -> Path:
    return Path(__file__).resolve().parents[1]

def mkdir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path, obj):
    path = Path(path)
    mkdir(path.parent)
    def conv(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, indent=2, default=conv), encoding='utf-8')


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


@dataclass(frozen=True)
class Settings:
    mode: str = 'full'
    cv_folds: int = 10
    n_tuning: int = 50
    generations: int = 100
    population: int = 10
    elite: int = 5
    cma_evals: int = 1000
    sigma: float = 0.3
    stagnation_generations: int = 20
    improvement_tolerance: float = 1e-4
    shap_background: int = 50
    shap_explained: int = 100
    kernel_shap_nsamples: int = 254
    n_jobs: int = 1


def settings(mode='full'):
    if mode == 'full': return Settings()
    if mode == 'smoke':
        return Settings(mode='smoke',cv_folds=2,n_tuning=2,generations=2,
                        cma_evals=20,shap_background=4,shap_explained=6,
                        kernel_shap_nsamples=18)
    raise ValueError('Use mode="smoke" or mode="full".')


def load_data(path=None, validate=True):
    path = Path(path) if path is not None else root() / 'data/processed/All_with_group.csv'
    df = pd.read_csv(path)
    required = [*FEATURES, 'group', 'makespan', 'normalized_makespan', 'performance']
    missing = sorted(set(required) - set(df.columns))
    if missing: raise ValueError(f'Missing columns: {missing}')
    if df[required].isna().any().any(): raise ValueError('Missing values in required fields.')
    if not np.isfinite(df[[*FEATURES,'makespan','normalized_makespan']].to_numpy()).all():
        raise ValueError('Nonfinite numerical values.')
    if not set(df.performance.unique()).issubset({0,1}): raise ValueError('Invalid class labels.')
    if validate:
        if len(df) != 24000 or df.group.nunique() != 800:
            raise ValueError('Unexpected dataset size; this code targets the 24,000-profile dataset.')
        if not df.groupby('group',sort=False).size().eq(30).all():
            raise ValueError('Every group must contain exactly 30 profiles.')
        if not df.group.eq(np.arange(len(df))//30 + 1).all():
            raise ValueError('Positional group order is not the reported original order.')
        if not np.allclose(df.makespan.to_numpy(), df[list(TIMES)].sum(axis=1).to_numpy(),atol=1e-10):
            raise ValueError('Makespan differs from its five input components.')
        reference = label_by_coefficient(df,1.5)
        if not np.array_equal(reference,df.performance.to_numpy()):
            raise ValueError('The stored normalised-makespan labels do not reproduce performance.')
    return df


def normalised_makespan(df,prefer_stored=True):
    """Preserve supplied full-precision normalised values when available.

    Recomputing with sums can flip threshold-adjacent rows via rounding.
    """
    if prefer_stored and 'normalized_makespan' in df:
        return pd.Series(df['normalized_makespan'].to_numpy(dtype=float),index=df.index)
    m = df['makespan'] if 'makespan' in df else df[list(TIMES)].sum(axis=1)
    low=m.groupby(df.group).transform('min'); high=m.groupby(df.group).transform('max')
    if high.eq(low).any(): raise ValueError('Zero within-group makespan range.')
    return (m-low)/(high-low)


def label_by_coefficient(df,coefficient=1.5):
    z = normalised_makespan(df)
    threshold = coefficient * z.groupby(df['group']).transform('median')
    return (z < threshold).astype(int).to_numpy()


def partitions(df,protocol='job',seed=42):
    idx = np.arange(len(df))
    if protocol=='job':
        train,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed)
                        .split(idx,df.performance,groups=df.group))
        assert set(df.group.iloc[train]).isdisjoint(df.group.iloc[test])
    elif protocol=='profile':
        train,test=train_test_split(idx,test_size=.2,random_state=seed,shuffle=True,stratify=None)
    else: raise ValueError(protocol)
    return np.asarray(train),np.asarray(test)


def store_split(df,protocol='job',seed=42):
    tr,te=partitions(df,protocol=protocol,seed=seed)
    out=mkdir(root()/'results/splits')
    np.savez_compressed(out/f'{protocol}_seed{seed}.npz',train=tr,test=te)
    write_json(out/f'{protocol}_seed{seed}.json',dict(protocol=protocol,seed=seed,
        train_profiles=len(tr),test_profiles=len(te),train_jobs=df.group.iloc[tr].nunique(),
        test_jobs=df.group.iloc[te].nunique(),job_overlap=len(set(df.group.iloc[tr])&set(df.group.iloc[te]))))
    return tr,te


def classifier(name,seed=42,params=None):
    params=dict(params or {})
    if name=='XGBoost':
        base=dict(n_estimators=100,max_depth=6,learning_rate=.3,min_child_weight=1,
                  subsample=1.0,reg_lambda=1.0,colsample_bytree=1.0,
                  colsample_bylevel=1.0,colsample_bynode=1.0,reg_alpha=0.0,
                  gamma=0.0,scale_pos_weight=1,objective='binary:logistic',
                  tree_method='hist',eval_metric='logloss',random_state=seed,n_jobs=1)
        base.update(params)
        return XGBClassifier(**base)
    if name=='SVM':
        base=dict(kernel='rbf',C=1.0,gamma=.125,class_weight=None)
        base.update(params)
        return SVC(**base)
    raise ValueError(name)


def param_distributions(name):
    if name=='XGBoost':
        return {'model__n_estimators':[100,200,300,500],
                'model__max_depth':[2,3,4,5,6],
                'model__learning_rate':loguniform(.01,.2),
                'model__min_child_weight':[1,3,5,10],
                'model__subsample':uniform(.7,.3),
                'model__reg_lambda':loguniform(.1,10)}
    return {'model__C':loguniform(1e-2,1e3),'model__gamma':loguniform(1e-3,10)}


def get_cv(y,seed,settings):
    return list(StratifiedKFold(n_splits=settings.cv_folds,shuffle=True,random_state=seed)
                .split(np.arange(len(y)),y))


def hyper_tune(name,X,y,seed,cfg):
    pipe=Pipeline([('scale',StandardScaler()),('model',classifier(name,seed))])
    s=RandomizedSearchCV(pipe,param_distributions(name),n_iter=cfg.n_tuning,
         scoring=make_scorer(f1_score,pos_label=0,zero_division=0),
         cv=get_cv(y,seed,cfg),random_state=seed,n_jobs=cfg.n_jobs,refit=True,
         return_train_score=False,error_score='raise')
    start=time.perf_counter();s.fit(X,y);elapsed=time.perf_counter()-start
    return {k.removeprefix('model__'):v for k,v in s.best_params_.items()},float(s.best_score_),elapsed


def metrics(y,pred):
    pr,re,f1,_=precision_recall_fscore_support(y,pred,labels=[0,1],average='macro',zero_division=0)
    p0,r0,f0,_=precision_recall_fscore_support(y,pred,labels=[0,1],average=None,zero_division=0)
    return dict(accuracy=accuracy_score(y,pred),macro_precision=pr,macro_recall=re,
                macro_f1=f1,class0_precision=p0[0],class0_recall=r0[0],class0_f1=f0[0],
                confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())


def fit_test(name,Xtrain,ytrain,Xtest,ytest,seed,params=None,scaled=True,weights=None):
    multiplier=np.ones(Xtrain.shape[1]) if weights is None else len(FEATURES)*np.asarray(weights)
    if scaled:
        scaler=StandardScaler().fit(Xtrain)
        tr=scaler.transform(Xtrain)*multiplier
        te=scaler.transform(Xtest)*multiplier
    else:
        if weights is not None: raise ValueError('Feature weighting requires standardisation.')
        scaler=None;tr=Xtrain.to_numpy();te=Xtest.to_numpy()
    m=classifier(name,seed,params)
    begin=time.perf_counter();m.fit(tr,ytrain);fit_sec=time.perf_counter()-begin
    begin=time.perf_counter();pred=m.predict(te);predict_sec=time.perf_counter()-begin
    return dict(model=m,scaler=scaler,weights=weights,
                metrics=metrics(ytest,pred),train_seconds=fit_sec,
                test_predict_seconds=predict_sec,predictions=np.asarray(pred))


class WeightFitness:
    def __init__(self,name,params,X,y,seed,cfg):
        self.name,self.params,self.seed,self.cfg=name,params,seed,cfg
        self.X=np.asarray(X,dtype=float);self.y=np.asarray(y,dtype=int)
        self.folds=get_cv(self.y,seed,cfg)
        self.calls=0
        self.best_loss=np.inf
        self.best_weights=None

    def __call__(self,weights):
        weights=np.asarray(weights,float)
        if len(weights)!=self.X.shape[1] or not np.all(weights>0) or not np.isclose(weights.sum(),1):
            raise ValueError('Weights must be strictly positive and sum to one.')
        mult=self.X.shape[1]*weights
        scores=[]
        for train,valid in self.folds:
            scaler=StandardScaler().fit(self.X[train])
            xt=scaler.transform(self.X[train])*mult
            xv=scaler.transform(self.X[valid])*mult
            model=classifier(self.name,self.seed,self.params)
            model.fit(xt,self.y[train])
            pred=model.predict(xv)
            scores.append(f1_score(self.y[valid],pred,pos_label=0,zero_division=0))
        loss=1-float(np.mean(scores))
        self.calls+=1
        if loss<self.best_loss:
            self.best_loss=loss;self.best_weights=weights.copy()
        return loss


def softmax(x):
    v=np.asarray(x,float);v=np.exp(v-np.max(v));return v/v.sum()


def shap_initial_weights(name,params,X,y,seed,cfg,archive_path=None):
    """Training-only SHAP, with explicit finite sampling for computational tractability.

    For XGBoost explains class-1 probability; negative gives class-0 attribution.
    For SVM explains negative decision_function (class-0 decision direction).
    Sampling choices should be disclosed in reproducibility documentation.
    """
    import shap
    X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=int)
    rng=np.random.default_rng(seed)
    bg_idx=np.sort(rng.choice(len(X),min(cfg.shap_background,len(X)),replace=False))
    ex_idx=np.sort(rng.choice(len(X),min(cfg.shap_explained,len(X)),replace=False))
    scaler=StandardScaler().fit(X)
    tr=scaler.transform(X);bg=tr[bg_idx];ex=tr[ex_idx]
    model=classifier(name,seed,params).fit(tr,y)
    start=time.perf_counter()
    if name=='XGBoost':
        explainer=shap.TreeExplainer(model,data=bg,feature_perturbation='interventional',
                                    model_output='probability')
        values=-np.asarray(explainer.shap_values(ex),dtype=float)
    else:
        explainer=shap.KernelExplainer(lambda z:-model.decision_function(z), bg)
        values=np.asarray(explainer.shap_values(ex,nsamples=cfg.kernel_shap_nsamples,
                           silent=True),dtype=float)
    if values.ndim!=2 or values.shape[1]!=len(FEATURES):
        raise ValueError(f'Unexpected SHAP array shape {values.shape}')
    importance=np.abs(values).mean(axis=0)
    w=(importance+1e-8)/np.sum(importance+1e-8)
    sec=time.perf_counter()-start
    if archive_path:
        mkdir(Path(archive_path).parent)
        np.savez_compressed(archive_path,values=values,importance=importance,initial_weights=w,
             background_train_positions=bg_idx,explained_train_positions=ex_idx,
             features=np.array(FEATURES),model=np.array(name),seed=np.array(seed))
    return w,sec


def optimize_cma(name,params,X,y,seed,cfg,initial_weights):
    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.callback import Callback
    from pymoo.optimize import minimize
    objective=WeightFitness(name,params,X,y,seed,cfg)
    class P(ElementwiseProblem):
        def __init__(self): super().__init__(n_var=len(FEATURES),n_obj=1)
        def _evaluate(self,v,out,*args,**kwargs):
            out['F']=objective(softmax(v))
    class StopOnStagnation(Callback):
        def __init__(self): super().__init__();self.best=np.inf;self.wait=0
        def notify(self,algorithm):
            if objective.best_loss < self.best-cfg.improvement_tolerance:
                self.best=objective.best_loss;self.wait=0
            else: self.wait+=1
            if self.wait>=cfg.stagnation_generations or objective.calls>=cfg.cma_evals:
                algorithm.termination.force_termination=True
    w=np.asarray(initial_weights,float)
    theta=np.log(w);theta-=theta.mean()
    algorithm=CMAES(x0=theta,sigma=cfg.sigma,normalize=False,
                    pop_size=cfg.population,CMA_mu=cfg.elite,
                    maxfevals=cfg.cma_evals,eval_initial_x=False)
    start=time.perf_counter()
    result=minimize(P(),algorithm,termination=('n_gen',cfg.generations),seed=seed,
                    callback=StopOnStagnation(),verbose=False)
    seconds=time.perf_counter()-start
    if objective.best_weights is None: raise RuntimeError('CMA-ES did not evaluate a candidate.')
    return dict(weights=objective.best_weights,loss=objective.best_loss,
                n_evaluations=objective.calls,search_seconds=seconds,
                raw_best_solution=None if result.X is None else np.asarray(result.X).tolist())


def optimize_random(name,params,X,y,seed,cfg,n_evaluations):
    objective=WeightFitness(name,params,X,y,seed,cfg)
    rng=np.random.default_rng(seed)
    begin=time.perf_counter()
    for _ in range(n_evaluations): objective(rng.dirichlet(np.ones(len(FEATURES))))
    return dict(weights=objective.best_weights,loss=objective.best_loss,
                n_evaluations=objective.calls,search_seconds=time.perf_counter()-begin)


def save_model(path,fit):
    path=Path(path);mkdir(path.parent)
    joblib.dump({k:fit[k] for k in ('model','scaler','weights')},path)


def environment_manifest(cfg=None):
    from importlib.metadata import version,PackageNotFoundError
    packages=['numpy','pandas','scipy','scikit-learn','xgboost','shap','pymoo','cma',
              'lime','matplotlib','joblib']
    versions={}
    for package in packages:
        try: versions[package]=version(package)
        except PackageNotFoundError:versions[package]=None
    d=dict(python=platform.python_version(),platform=platform.platform(),packages=versions)
    if cfg:d['settings']=asdict(cfg)
    return d


def summarize_trials(csv_path,group_cols,metric_cols):
    f=pd.read_csv(csv_path)
    agg=f.groupby(group_cols)[metric_cols].agg(['mean','std'])
    agg.columns=['_'.join(parts) for parts in agg.columns]
    return agg.reset_index()


def select_mode(df,mode='full',n_smoke_jobs=20):
    if mode=='full': return df.reset_index(drop=True)
    if mode!='smoke': raise ValueError(mode)
    jobs=np.array(sorted(df.group.unique()))
    picked=np.random.default_rng(42).choice(jobs,size=min(n_smoke_jobs,len(jobs)),replace=False)
    return df.loc[df.group.isin(picked)].reset_index(drop=True)


def result_folder(experiment,mode='full'):
    return mkdir(root()/'results'/mode/experiment)


def tuned_path(mode,name,seed):
    return result_folder('tuning',mode)/f'{name}_seed{seed}.json'


def read_tuned(mode,name,seed):
    path=tuned_path(mode,name,seed)
    if not path.exists():
        raise FileNotFoundError(f'Missing tuning parameters: {path}')
    return json.loads(path.read_text(encoding='utf-8'))['params']
