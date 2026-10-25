"""Synthetic NEWS data fidelity."""

from pathlib import Path
from datetime import datetime
from importlib.metadata import version
from itertools import combinations
import hashlib
import json
import os
import platform
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance, rankdata
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = PROJECT / "data" / "news" / "NEWSsample.csv"
GENERATED_PATH = PROJECT / "data" / "news" / "NEWSgenerated.csv"
GENERATED_RENAME = {}
CSV_SEPARATOR = ","
CSV_ENCODING = "utf-8-sig"
MISSING_TOKENS = ["", "NA", "N/A", "NULL", "null", "NaN", "nan"]

NUMERIC_COLUMNS = ["pulse_value", "temp_value", "bp_sys_value",
                   "bp_dia_value", "resp_rate_value", "o2_sats_value"]
CATEGORICAL_COLUMNS = ["sex"]
SCORE_COLUMN = "ews_score"
UNIT_COLUMN = None         # Documented repeated-observation identifier, if available.
MEASUREMENT_SEMANTICS_CONFIRMED = False
REFERENCE_PROVENANCE = "Supplied reference example; clinical provenance not independently verified"
REFERENCE_USED_IN_GENERATION = True
DOCUMENTED_RANGES = {}     # e.g. {column: (documented_lower, documented_upper)}

MODE = os.environ.get("NEWS_ANALYSIS_MODE", "full")
SEED = 42
MAX_STORED_ROWS = 20000
CHUNK_SIZE = 50000
MAX_ASSOC_ROWS = 2000
MAX_DISTANCE_ROWS = 350
MAX_MMD_ROWS = 500
MAX_SOURCE_ROWS_PER_DATASET = 1000
BOOTSTRAPS = 1000 if MODE == "full" else 40
SOURCE_REPETITIONS = 5 if MODE == "full" else 1
N_TREES = 150 if MODE == "full" else 30
N_JOBS = 1  # Number of parallel workers.
assert MODE in {"quick", "full"}
assert len(NUMERIC_COLUMNS) >= 2 and len(set(NUMERIC_COLUMNS)) == len(NUMERIC_COLUMNS)
assert not set(NUMERIC_COLUMNS) & set(CATEGORICAL_COLUMNS)
OUT = PROJECT / "results" / "news_fidelity" / (MODE + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
OUT.mkdir(parents=True, exist_ok=False)
(OUT / "figures").mkdir()
(OUT / "tables").mkdir()
print("Output directory:", OUT.as_posix())


def save_table(frame, name, index=False):
    frame.to_csv(OUT / "tables" / (name + ".csv"), index=index)
    return frame

def save_figure(fig, name):
    fig.savefig(OUT / "figures" / (name + ".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "figures" / (name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()

def clean_headers(columns):
    cleaned = [str(c).strip() for c in columns]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("Duplicate column names after stripping spaces.")
    return cleaned

def scan_file(path, rename, seed):
    if not path.exists():
        raise FileNotFoundError(f"Provide the NEWS input in data/news/ or update its path: {path.resolve()}")
    raw_header = pd.read_csv(path, sep=CSV_SEPARATOR, encoding=CSV_ENCODING, nrows=0).columns
    header = clean_headers(raw_header)
    header = [rename.get(c, c) for c in header]
    if len(header) != len(set(header)):
        raise ValueError("Column renaming created duplicate names.")
    required = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ([UNIT_COLUMN] if UNIT_COLUMN else [])
    absent = sorted(set(required) - set(header))
    if absent:
        raise ValueError(f"{path.name}: missing selected columns {absent}. Available: {header}")
    numeric_to_check = NUMERIC_COLUMNS + ([SCORE_COLUMN] if SCORE_COLUMN in header and SCORE_COLUMN not in NUMERIC_COLUMNS else [])
    missing_count = dict.fromkeys(header, 0)
    invalid_count = dict.fromkeys(numeric_to_check, 0)
    total = 0
    reservoir = None
    generator = np.random.default_rng(seed)
    reader = pd.read_csv(path, sep=CSV_SEPARATOR, encoding=CSV_ENCODING,
                         dtype=str, chunksize=CHUNK_SIZE, keep_default_na=False)
    for chunk in reader:
        chunk.columns = header
        for col in header:
            chunk[col] = chunk[col].str.strip().replace(MISSING_TOKENS, np.nan)
            missing_count[col] += int(chunk[col].isna().sum())
        for col in numeric_to_check:
            parsed = pd.to_numeric(chunk[col], errors="coerce")
            invalid_count[col] += int((chunk[col].notna() & (~np.isfinite(parsed))).sum())
        # Keep original zero-based CSV data-row positions separately.
        chunk["__source_position__"] = np.arange(total, total + len(chunk))
        chunk["__priority__"] = generator.random(len(chunk))
        total += len(chunk)
        reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
        if len(reservoir) > MAX_STORED_ROWS:
            reservoir = reservoir.nsmallest(MAX_STORED_ROWS, "__priority__")
    if reservoir is None or total == 0:
        raise ValueError(f"{path.name} is empty.")
    reservoir = reservoir.drop(columns="__priority__").sort_values("__source_position__").reset_index(drop=True)
    audit = pd.DataFrame({"column":header, "file_rows":total,
                          "missing_count":[missing_count[c] for c in header],
                          "numeric_parse_or_nonfinite_count":[invalid_count.get(c, np.nan) for c in header]})
    audit["missing_fraction"] = audit["missing_count"] / total
    for col in numeric_to_check:
        reservoir[col] = pd.to_numeric(reservoir[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return reservoir, audit, total, header

reference, ref_audit, n_ref_full, ref_header = scan_file(REFERENCE_PATH, {}, SEED)
generated, gen_audit, n_gen_full, gen_header = scan_file(GENERATED_PATH, GENERATED_RENAME, SEED+1)
save_table(ref_audit, "reference_full_file_audit")
save_table(gen_audit, "generated_full_file_audit")
save_table(pd.DataFrame({"column": sorted(set(ref_header)|set(gen_header))}).assign(
    in_reference=lambda d:d.column.isin(ref_header), in_generated=lambda d:d.column.isin(gen_header)), "schema_alignment")
for name, data, total in [("Reference",reference,n_ref_full),("Generated",generated,n_gen_full)]:
    print(f"{name}: {total:,} file rows, {len(data):,} retained rows")
print("Reference-only columns:", sorted(set(ref_header)-set(gen_header)))
print("Generated-only columns:", sorted(set(gen_header)-set(ref_header)))

print("Scope:", "Verified measurement mapping" if MEASUREMENT_SEMANTICS_CONFIRMED else "Encoded-column fidelity; clinical interpretation not established")

manifest = {"reference_path":str(REFERENCE_PATH.resolve()),"generated_path":str(GENERATED_PATH.resolve()),
    "reference_sha256":sha256(REFERENCE_PATH),"generated_sha256":sha256(GENERATED_PATH),
    "mode":MODE,"seed":SEED,"reference_rows":n_ref_full,"generated_rows":n_gen_full,
    "reference_retained":len(reference),"generated_retained":len(generated),
    "numeric_columns":NUMERIC_COLUMNS,"categorical_columns":CATEGORICAL_COLUMNS,
    "score_column":SCORE_COLUMN,"unit_column":UNIT_COLUMN,"generated_rename":GENERATED_RENAME,
    "measurement_semantics_confirmed":MEASUREMENT_SEMANTICS_CONFIRMED,
    "reference_provenance":REFERENCE_PROVENANCE,"reference_used_in_generation":REFERENCE_USED_IN_GENERATION,
    "documented_ranges":DOCUMENTED_RANGES,"bootstrap_draws":BOOTSTRAPS,
    "source_repetitions":SOURCE_REPETITIONS,"max_stored_rows":MAX_STORED_ROWS,
    "max_association_rows":MAX_ASSOC_ROWS,"max_distance_rows":MAX_DISTANCE_ROWS,
    "max_mmd_rows":MAX_MMD_ROWS,"max_source_rows":MAX_SOURCE_ROWS_PER_DATASET,
    "source_trees":N_TREES,"n_jobs":N_JOBS,"python":platform.python_version(),
    "versions":{p:version(p) for p in ["numpy","pandas","scipy","scikit-learn","matplotlib","seaborn"]}}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
for name,data in [("reference",reference),("generated",generated)]:
    save_table(data[["__source_position__"]], name+"_retained_positions")


rows=[]
for source,data in [("Reference",reference),("Generated",generated)]:
    for col in NUMERIC_COLUMNS:
        a=data[col].dropna()
        lo,hi=DOCUMENTED_RANGES.get(col,(None,None))
        rows.append({"source":source,"column":col,"n_valid":len(a),"n_unique":a.nunique(),
                     "min":a.min(),"max":a.max(),"mean":a.mean(),"sd":a.std(ddof=1),
                     "median":a.median(),"q25":a.quantile(.25),"q75":a.quantile(.75),
                     "documented_low":lo,"documented_high":hi,
                     "outside_documented_range_fraction":float(((a<lo)|(a>hi)).mean()) if lo is not None and hi is not None and len(a) else np.nan})
quality=save_table(pd.DataFrame(rows),"selected_column_quality")

constant_ref=[c for c in ref_header if reference[c].nunique(dropna=True)<=1]
print("Constant/empty columns in retained reference:",constant_ref)

def canonical_rows(data, columns):
    d=data[columns].copy()
    for c in columns:
        d[c]=d[c].astype("string").fillna("<MISSING>")
    return pd.util.hash_pandas_object(d,index=False)

selected=NUMERIC_COLUMNS+CATEGORICAL_COLUMNS
common_content=[c for c in ref_header if c in gen_header and not c.lower().endswith("_id") and c!=UNIT_COLUMN]
overlap_rows=[]
for definition,cols in [("selected_analysis_fields",selected),("common_non_id_content",common_content)]:
    rh=canonical_rows(reference,cols); gh=canonical_rows(generated,cols)
    overlap_rows.append({"definition":definition,"n_fields":len(cols),
        "reference_duplicate_fraction":rh.duplicated().mean(),
        "generated_duplicate_fraction":gh.duplicated().mean(),
        "generated_rows_matching_reference_fraction":gh.isin(set(rh)).mean()})
overlap=save_table(pd.DataFrame(overlap_rows),"retained_sample_overlap")

preview_corr=reference[NUMERIC_COLUMNS].corr(method="spearman")
offdiag=preview_corr.to_numpy()[np.triu_indices(len(NUMERIC_COLUMNS),1)]
print("Reference maximum absolute off-diagonal Spearman:", np.nanmax(np.abs(offdiag)))
if not MEASUREMENT_SEMANTICS_CONFIRMED:
    print("Measurement units and coding are unverified; results describe recorded fields.")


marginal_rows=[]
for c in NUMERIC_COLUMNS:
    a=reference[c].dropna().to_numpy(); b=generated[c].dropna().to_numpy()
    if min(len(a),len(b))<3:
        raise ValueError(f"Insufficient valid values in {c}.")
    scale=np.quantile(a,.75)-np.quantile(a,.25)
    if scale<=0: scale=np.std(a,ddof=1)
    w=wasserstein_distance(a,b)
    marginal_rows.append({"column":c,"reference_n":len(a),"generated_n":len(b),
        "ks_distance":ks_2samp(a,b,method="asymp").statistic,
        "wasserstein_native":w,"reference_scale":scale,
        "wasserstein_normalized":w/scale if scale>0 else np.nan,
        "generated_outside_reference_range_fraction":np.mean((b<a.min())|(b>a.max())),
        "median_difference":np.median(b)-np.median(a)})
marginals=save_table(pd.DataFrame(marginal_rows),"marginal_numeric_fidelity")

fig,axes=plt.subplots(int(np.ceil(len(NUMERIC_COLUMNS)/3)),3,figsize=(13,3.3*np.ceil(len(NUMERIC_COLUMNS)/3)),squeeze=False)
for ax,c in zip(axes.flat,NUMERIC_COLUMNS):
    for data,color,label in [(reference,"#007F86","Reference"),(generated,"#C45A31","Generated")]:
        vals=np.sort(data[c].dropna().to_numpy()); ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where="post",color=color,label=label,lw=1.5)
    ax.set(title=c,xlabel="Recorded value (units unverified)" if not MEASUREMENT_SEMANTICS_CONFIRMED else "Value",ylabel="Empirical CDF")
for ax in list(axes.flat)[len(NUMERIC_COLUMNS):]:ax.set_visible(False)
axes.flat[0].legend(); fig.tight_layout(); save_figure(fig,"01_marginal_ecdfs")
q=np.linspace(.01,.99,99)
fig,ax=plt.subplots(figsize=(9,5))
for c in NUMERIC_COLUMNS:
    a=reference[c].dropna(); b=generated[c].dropna()
    scale=marginals.set_index("column").loc[c,"reference_scale"]
    if scale>0:ax.plot(q,(np.quantile(b,q)-np.quantile(a,q))/scale,label=c)
ax.axhline(0,color="black",lw=.8); ax.set(xlabel="Quantile",ylabel="Quantile difference / reference scale",title="Marginal quantile differences")
ax.legend(bbox_to_anchor=(1.02,1),loc="upper left"); fig.tight_layout(); save_figure(fig,"02_quantile_difference")


category_rows=[]
cat_columns=CATEGORICAL_COLUMNS+([SCORE_COLUMN] if SCORE_COLUMN in reference and SCORE_COLUMN in generated and SCORE_COLUMN not in CATEGORICAL_COLUMNS else [])
for c in cat_columns:
    a=reference[c].astype("string").fillna("<MISSING>")
    b=generated[c].astype("string").fillna("<MISSING>")
    levels=sorted(set(a)|set(b))
    pa=a.value_counts(normalize=True).reindex(levels,fill_value=0)
    pb=b.value_counts(normalize=True).reindex(levels,fill_value=0)
    category_rows.append({"column":c,"n_categories":len(levels),"total_variation":.5*np.abs(pa-pb).sum(),
                          "js_divergence_bits":float(jensenshannon(pa,pb,base=2)**2)})
    save_table(pd.DataFrame({"category":levels,"reference_probability":pa.values,"generated_probability":pb.values}),"category_"+c)
    if len(levels)<=30:
        fig,ax=plt.subplots(figsize=(max(6,len(levels)*.45),4))
        x=np.arange(len(levels)); ax.bar(x-.2,pa,width=.4,label="Reference",color="#007F86"); ax.bar(x+.2,pb,width=.4,label="Generated",color="#C45A31")
        ax.set_xticks(x,levels); ax.set(ylabel="Proportion",title=f"Recorded {c} distribution"); ax.legend();fig.tight_layout();save_figure(fig,"03_category_"+c)
categories=save_table(pd.DataFrame(category_rows),"categorical_fidelity")

print("NEWS scoring-rule consistency was not assessed.")


def select_units(data,seed):
    if UNIT_COLUMN is None:return data.copy()
    if data[UNIT_COLUMN].isna().any():raise ValueError("Missing unit identifiers must be resolved first.")
    return data.sample(frac=1,random_state=seed).drop_duplicates(UNIT_COLUMN).copy()
ref_units=select_units(reference,SEED+2)
gen_units=select_units(generated,SEED+3)
ref_complete=ref_units.dropna(subset=NUMERIC_COLUMNS)
gen_complete=gen_units.dropna(subset=NUMERIC_COLUMNS)
print("Complete cases (reference/generated):",len(ref_complete),len(gen_complete))
print("Excluded incomplete units:",len(ref_units)-len(ref_complete),len(gen_units)-len(gen_complete))
if min(len(ref_complete),len(gen_complete))<20:raise ValueError("Too few complete cases for dependence analysis.")
ref_assoc=ref_complete.sample(n=min(MAX_ASSOC_ROWS,len(ref_complete)),random_state=SEED+4)
gen_assoc=gen_complete.sample(n=min(MAX_ASSOC_ROWS,len(gen_complete)),random_state=SEED+5)
active=[c for c in NUMERIC_COLUMNS if ref_assoc[c].nunique()>1 and gen_assoc[c].nunique()>1]
print("Excluded constant columns:",sorted(set(NUMERIC_COLUMNS)-set(active)))
if len(active)<2:raise ValueError("At least two nonconstant columns in both sources are required.")
save_table(ref_assoc[["__source_position__"]],"reference_association_positions")
save_table(gen_assoc[["__source_position__"]],"generated_association_positions")
A=ref_assoc[active].to_numpy(float); B=gen_assoc[active].to_numpy(float)
PAIRS=list(combinations(range(len(active)),2))
TRIU=np.triu_indices(len(active),1)

def correlation_matrix(array,method):
    z=rankdata(array,axis=0,method="average") if method=="spearman" else array
    with np.errstate(invalid="ignore",divide="ignore"):
        return np.corrcoef(z,rowvar=False)

correlation_summary=[]
correlations={}
for method in ["pearson","spearman"]:
    cr=correlation_matrix(A,method); cg=correlation_matrix(B,method); delta=cg-cr
    correlations[method]=(cr,cg)
    for name,mat in [("reference",cr),("generated",cg),("difference",delta)]:
        save_table(pd.DataFrame(mat,index=active,columns=active),method+"_"+name,index=True)
    correlation_summary.append({"method":method,"n_reference":len(A),"n_generated":len(B),
        "mean_absolute_pair_difference":np.nanmean(np.abs(delta[TRIU])),
        "root_mean_square_pair_difference":np.sqrt(np.nanmean(delta[TRIU]**2)),
        "max_absolute_pair_difference":np.nanmax(np.abs(delta[TRIU])),
        "reference_max_absolute_pair_correlation":np.nanmax(np.abs(cr[TRIU]))})
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for ax,mat,title,limit in zip(axes,[cr,cg,delta],["Reference","Generated","Generated minus reference"],[1,1,max(.1,float(np.nanmax(abs(delta))))]):
        sns.heatmap(pd.DataFrame(mat,index=active,columns=active),ax=ax,cmap="vlag",center=0,vmin=-limit,vmax=limit,annot=True,fmt=".2f",square=True,cbar_kws={"shrink":.7})
        ax.set_title(title);ax.tick_params(axis="x",rotation=65)
    fig.suptitle(method.capitalize()+" dependence: selected recorded columns",y=1.02);fig.tight_layout();save_figure(fig,"04_"+method+"_dependence")
correlation_summary=save_table(pd.DataFrame(correlation_summary),"dependence_matrix_summary")

rng=np.random.default_rng(SEED+20)
r_boot=np.full((BOOTSTRAPS,len(PAIRS)),np.nan)
g_boot=np.full_like(r_boot,np.nan)
for k in range(BOOTSTRAPS):
    r_boot[k]=correlation_matrix(A[rng.integers(0,len(A),len(A))],"spearman")[TRIU]
    g_boot[k]=correlation_matrix(B[rng.integers(0,len(B),len(B))],"spearman")[TRIU]
    if (k+1)%max(1,BOOTSTRAPS//5)==0:print(f"Bootstrap {k+1}/{BOOTSTRAPS}",flush=True)
cr,cg=correlations["spearman"]
ci_rows=[]
for k,(i,j) in enumerate(PAIRS):
    rr=r_boot[:,k];gg=g_boot[:,k];dd=gg-rr
    valid=np.isfinite(dd)
    if valid.sum()<.9*BOOTSTRAPS:
        warnings.warn(f"Unstable bootstrap for {active[i]} / {active[j]}; inspect valid draw count.")
    def interval(v):
        v=v[np.isfinite(v)]
        return np.quantile(v,[.025,.975]) if len(v) else [np.nan,np.nan]
    rl,rh=interval(rr);gl,gh=interval(gg);dl,dh=interval(dd)
    ci_rows.append({"feature_1":active[i],"feature_2":active[j],"reference_rho":cr[i,j],
        "reference_lower":rl,"reference_upper":rh,"generated_rho":cg[i,j],
        "generated_lower":gl,"generated_upper":gh,"delta_rho":cg[i,j]-cr[i,j],
        "delta_lower":dl,"delta_upper":dh,"valid_difference_draws":int(valid.sum())})
ci=save_table(pd.DataFrame(ci_rows),"spearman_pointwise_bootstrap_intervals")

plot_ci=ci.sort_values("delta_rho").reset_index(drop=True)
fig,ax=plt.subplots(figsize=(10,max(5,.38*len(plot_ci))))
y=np.arange(len(plot_ci));ax.hlines(y,plot_ci.delta_lower,plot_ci.delta_upper,color="#007F86",lw=2)
ax.scatter(plot_ci.delta_rho,y,color="#C45A31",s=24,zorder=3)
ax.axvline(0,color="black",ls="--",lw=1)
ax.set_yticks(y,plot_ci.feature_1+" / "+plot_ci.feature_2)
ax.set(xlabel="Generated minus reference Spearman rho",title="Pointwise 95% conditional bootstrap intervals")
fig.tight_layout();save_figure(fig,"05_correlation_difference_intervals")


def centered_distances(v):
    d=np.abs(v[:,None]-v[None,:])
    return d-d.mean(axis=0,keepdims=True)-d.mean(axis=1,keepdims=True)+d.mean()

def distance_corr_matrix(arr):
    centered=[centered_distances(arr[:,j]) for j in range(arr.shape[1])]
    norms=np.array([np.mean(d*d) for d in centered])
    m=np.eye(arr.shape[1])
    for i,j in combinations(range(arr.shape[1]),2):
        denominator=np.sqrt(norms[i]*norms[j])
        value=np.sqrt(max(0,float(np.mean(centered[i]*centered[j])))/denominator) if denominator>0 else np.nan
        m[i,j]=m[j,i]=value
    return m

rng=np.random.default_rng(SEED+30)
n=min(MAX_DISTANCE_ROWS,len(A),len(B))
DA=A[rng.choice(len(A),n,replace=False)]; DB=B[rng.choice(len(B),n,replace=False)]
dcr=distance_corr_matrix(DA); dcg=distance_corr_matrix(DB)
for name,mat in [("reference",dcr),("generated",dcg)]:
    save_table(pd.DataFrame(mat,index=active,columns=active),"distance_correlation_"+name,index=True)
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,mat,title in zip(axes,[dcr,dcg],["Reference","Generated"]):
    sns.heatmap(pd.DataFrame(mat,index=active,columns=active),ax=ax,vmin=0,vmax=1,cmap="crest",annot=True,fmt=".2f",square=True)
    ax.set_title(f"{title}: distance correlation (n={n})");ax.tick_params(axis="x",rotation=65)
fig.tight_layout();save_figure(fig,"06_nonlinear_dependence")

n=min(MAX_MMD_ROWS,len(A),len(B))
MA=A[rng.choice(len(A),n,replace=False)];MB=B[rng.choice(len(B),n,replace=False)]
center=np.median(MA,axis=0);scale=np.quantile(MA,.75,axis=0)-np.quantile(MA,.25,axis=0)
scale=np.where(scale>0,scale,1)
Z=np.vstack([(MA-center)/scale,(MB-center)/scale])
d2=cdist(Z,Z,metric="sqeuclidean");positive=d2[np.triu_indices(len(Z),1)];positive=positive[positive>0]
bandwidth=float(np.median(positive)) if len(positive) else 1.0
K=np.exp(-d2/(2*bandwidth))

def mmd_stat(kernel,idx_a,idx_b):
    return float(kernel[np.ix_(idx_a,idx_a)].mean()+kernel[np.ix_(idx_b,idx_b)].mean()-2*kernel[np.ix_(idx_a,idx_b)].mean())
observed_mmd=max(0.,mmd_stat(K,np.arange(n),np.arange(n,2*n)))
null=[]
for _ in range(199 if MODE=="full" else 39):
    order=rng.permutation(2*n);null.append(max(0.,mmd_stat(K,order[:n],order[n:])))
mmd_summary=save_table(pd.DataFrame([{"n_each":n,"biased_mmd_squared":observed_mmd,
    "kernel_bandwidth_squared":bandwidth,"pooled_random_allocation_median":np.median(null),
    "pooled_random_allocation_q95":np.quantile(null,.95),"inferential_p_value_reported":False}]),"mmd_diagnostic")

fig,ax=plt.subplots(figsize=(7,4));ax.hist(null,bins=20,color="#91BFC1",edgecolor="white")
ax.axvline(observed_mmd,color="#C45A31",lw=2,label="Observed")
ax.set(xlabel="Biased MMD squared",ylabel="Random allocations",title="Pooled random-allocation diagnostic (not an inferential test)")
ax.legend();fig.tight_layout();save_figure(fig,"07_mmd_diagnostic")


source_features=active+CATEGORICAL_COLUMNS
numeric_pipeline=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])
transformers=[("numeric",numeric_pipeline,active)]
if CATEGORICAL_COLUMNS:
    categorical_pipeline=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),
                                   ("encode",OneHotEncoder(handle_unknown="ignore"))])
    transformers.append(("categorical",categorical_pipeline,CATEGORICAL_COLUMNS))
preprocessor=ColumnTransformer(transformers)
source_models={
    "Logistic regression":Pipeline([("preprocess",clone(preprocessor)),("classifier",LogisticRegression(max_iter=2000,random_state=SEED))]),
    "Random forest":Pipeline([("preprocess",clone(preprocessor)),("classifier",RandomForestClassifier(n_estimators=N_TREES,min_samples_leaf=5,max_features="sqrt",n_jobs=N_JOBS,random_state=SEED))])}
auc_rows=[];roc_rows=[]
for repeat in range(SOURCE_REPETITIONS):
    n=min(MAX_SOURCE_ROWS_PER_DATASET,len(ref_units),len(gen_units))
    if n<25:raise ValueError("Insufficient rows for source classification.")
    r=ref_units.sample(n=n,random_state=SEED+100+repeat)
    g=gen_units.sample(n=n,random_state=SEED+200+repeat)
    xx=pd.concat([r[source_features],g[source_features]],ignore_index=True)
    for c in CATEGORICAL_COLUMNS:xx[c]=xx[c].astype(object).where(xx[c].notna(),np.nan)
    yy=np.r_[np.zeros(n,dtype=int),np.ones(n,dtype=int)]
    duplicate_group=canonical_rows(xx,source_features).to_numpy()
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED+repeat)
    folds=list(splitter.split(xx,yy,groups=duplicate_group))
    for train,test in folds:
        if len(np.unique(yy[train]))<2 or len(np.unique(yy[test]))<2:
            raise ValueError("Source CV fold has only one class; duplicate concentration prevents this CV design.")
    for name,estimator in source_models.items():
        pred=np.full(len(xx),np.nan)
        for train,test in folds:
            fitted=clone(estimator).fit(xx.iloc[train],yy[train])
            pred[test]=fitted.predict_proba(xx.iloc[test])[:,1]
        auc=roc_auc_score(yy,pred)
        auc_rows.append({"repetition":repeat+1,"classifier":name,"n_each":n,"oof_auc":auc})
        fpr,tpr,_=roc_curve(yy,pred)
        roc_rows.append((name,repeat,fpr,tpr))
        save_table(pd.DataFrame({"source":yy,"source_row":np.r_[r.__source_position__.to_numpy(),g.__source_position__.to_numpy()],"oof_probability_generated":pred}),f"source_predictions_{name.replace(' ','_')}_{repeat+1}")
        print(f"Source diagnostic {repeat+1}/{SOURCE_REPETITIONS}, {name}: OOF AUC={auc:.4f}",flush=True)
auc_results=save_table(pd.DataFrame(auc_rows),"source_classifier_auc")
auc_summary=auc_results.groupby("classifier").oof_auc.agg(["mean","std","min","max"])
save_table(auc_summary,"source_classifier_auc_summary",index=True)

fig,ax=plt.subplots(figsize=(6,5));colors={"Logistic regression":"#007F86","Random forest":"#C45A31"}
for name,repeat,fpr,tpr in roc_rows:ax.plot(fpr,tpr,color=colors[name],alpha=.45,lw=1.3,label=name if repeat==0 else None)
ax.plot([0,1],[0,1],"k--",lw=1);ax.set(xlabel="False positive rate",ylabel="True positive rate",title="Out-of-fold source discrimination")
ax.legend();fig.tight_layout();save_figure(fig,"08_source_discrimination")


scaler=StandardScaler().fit(A)
Ar=scaler.transform(A);Br=scaler.transform(B)
pca=PCA(n_components=2).fit(Ar)
pr=pca.transform(Ar);pg=pca.transform(Br)
fig,ax=plt.subplots(figsize=(7,5))
ax.scatter(pg[:,0],pg[:,1],s=9,alpha=.15,color="#C45A31",label="Generated")
ax.scatter(pr[:,0],pr[:,1],s=10,alpha=.35,color="#007F86",label="Reference")
ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} reference variance)",ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} reference variance)",title="Reference-fitted PCA projection")
ax.legend();fig.tight_layout();save_figure(fig,"09_pca_projection")
strongest=np.argsort(-np.abs(correlations["spearman"][0][TRIU]))[:min(3,len(PAIRS))]
fig,axes=plt.subplots(len(strongest),2,figsize=(11,3.5*len(strongest)),squeeze=False)
for row,k in enumerate(strongest):
    i,j=PAIRS[k]
    low=np.minimum(A[:,[i,j]].min(axis=0),B[:,[i,j]].min(axis=0));high=np.maximum(A[:,[i,j]].max(axis=0),B[:,[i,j]].max(axis=0))
    extent=(low[0],high[0],low[1],high[1])
    for ax,data,name in zip(axes[row],[A,B],["Reference","Generated"]):
        im=ax.hexbin(data[:,i],data[:,j],gridsize=22,mincnt=1,extent=extent,cmap="viridis",bins="log")
        ax.set(xlabel=active[i],ylabel=active[j],title=name);fig.colorbar(im,ax=ax,label="Count (log colour scale)")
fig.tight_layout();save_figure(fig,"10_joint_distributions")


spear=correlation_summary.set_index("method").loc["spearman"]
report=f"""# Statistical Fidelity of Synthetic NEWS Data

Analysis mode: {MODE}. Reference file rows: {n_ref_full:,}; generated file rows: {n_gen_full:,}.
Retained samples: {len(reference):,} reference and {len(generated):,} generated rows.
Selected numeric fields: {', '.join(NUMERIC_COLUMNS)}.
Measurement semantics verified: {MEASUREMENT_SEMANTICS_CONFIRMED}.
Reference provenance: {REFERENCE_PROVENANCE}.
Reference used in generation: {REFERENCE_USED_IN_GENERATION}.

## Pairwise dependence
Complete-case dependence analysis used {len(A):,} reference and {len(B):,} generated observations.
The mean absolute difference between pairwise Spearman correlations was {spear['mean_absolute_pair_difference']:.4f};
the largest absolute difference was {spear['max_absolute_pair_difference']:.4f}.
The largest absolute pairwise Spearman correlation in the reference was {spear['reference_max_absolute_pair_correlation']:.4f}.
Pair-specific point estimates and pointwise conditional 95% bootstrap intervals are provided in spearman_pointwise_bootstrap_intervals.csv.
These intervals describe empirical resampling variability and do not establish independent clinical validation or equivalence.

## Marginals and multivariate diagnostics
Numeric marginal effect sizes appear in marginal_numeric_fidelity.csv; categorical differences appear in categorical_fidelity.csv.
The Gaussian-kernel biased MMD-squared statistic was {observed_mmd:.6f} using {int(mmd_summary.iloc[0]['n_each'])} rows per source.
The pooled random-allocation distribution is descriptive; no inferential p-value was reported.
Source-classifier OOF AUC summaries (repetition SD/range, not population confidence intervals):
{auc_summary.to_string()}

## Interpretation limits
The analysis assesses statistical fidelity to a reference sample that informed generation.
The weak reference correlations limit conclusions about preservation of physiological dependence.
Clinical provenance, observation identity, measurement units, and coding remain unverified.
These diagnostics do not establish clinical realism, distributional equivalence, privacy, or NEWS scoring correctness.

## Reproducibility
See manifest.json, retained/association row-position files, numerical CSV outputs and PDF/PNG figures.
The input CSV files were not modified.
"""
(OUT/"interpretation_report.md").write_text(report,encoding="utf-8")
print(report)
print("Completed. Results:", OUT.as_posix())
