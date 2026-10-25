"""Runtime and statistical analysis."""

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


from itertools import product
import numpy as np,pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
base=r.result_folder('ablation',MODE)
file=base/'repetition_results.csv'
if not file.exists():raise FileNotFoundError(f'Missing ablation metrics: {file}')
df=pd.read_csv(file)
assert df[['model','seed','configuration']].drop_duplicates().shape[0]==len(df)
metrics=['accuracy','macro_precision','macro_recall','macro_f1']
summary=r.summarize_trials(file,['model','configuration'],metrics)
summary.to_csv(base/'paper_ablation_summary.csv',index=False)
print(summary.to_string(index=False))


# Paired exact sign-assignment Wilcoxon test.
def exact_wilcoxon_paired(a, b):
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    diff = diff[~np.isclose(diff, 0, atol=1e-14, rtol=0)]
    if len(diff) == 0:
        return 1.0
    ranks = rankdata(np.abs(diff), method='average')
    center = float(ranks.sum()) / 2
    observed = abs(float(ranks[diff > 0].sum()) - center)
    values = []
    for signs in product([-1, 1], repeat=len(diff)):
        signs = np.asarray(signs)
        values.append(abs(float(ranks[signs > 0].sum()) - center))
    return float(np.mean(np.asarray(values) >= observed - 1e-12))

subset = df.query('model == "SVM"').pivot(index='seed', columns='configuration', values='macro_f1')
if MODE == 'full' and set(subset.index) != set(r.SEEDS):
    raise ValueError('Incomplete full-mode SVM repetitions: all five seeds are required.')
comparisons = [('A2', 'A3'), ('A3', 'A4')]
pvals = [exact_wilcoxon_paired(subset[a], subset[b]) for a, b in comparisons]
p_adjusted = multipletests(pvals, method='holm')[1]
tests = pd.DataFrame({
    'comparison': [f'{a} vs {b}' for a, b in comparisons],
    'p_exact': pvals, 'p_holm': p_adjusted,
    'n_pairs': len(subset), 'mode': MODE
})
tests.to_csv(base / 'wilcoxon_comparisons.csv', index=False)
print(tests.to_string(index=False))


per_fit = df.groupby(['model', 'configuration'])[
    ['train_seconds', 'test_predict_seconds']
].agg(['mean', 'std'])
per_fit.to_csv(base / 'per_fit_runtime.csv')

records = []
for name in r.MODEL_NAMES:
    for seed in SEEDS:
        row = df.loc[(df.model == name) & (df.seed == seed) &
                     (df.configuration == 'A4')]
        if len(row) != 1:
            raise ValueError(f'Missing or duplicate A4 record for {name}, seed {seed}.')
        run = row.iloc[0]
        params_record = r.tuned_path(MODE, name, seed)
        if not params_record.exists():
            raise FileNotFoundError(f'Missing tuning timing: {params_record}')
        import json
        tuning = json.loads(params_record.read_text(encoding='utf-8'))
        seconds = {
            'tuning_seconds': float(tuning['search_seconds']),
            'shap_seconds': float(run['shap_seconds']),
            'optimise_seconds': float(run['optimise_seconds']),
            'final_fit_seconds': float(run['train_seconds'])
        }
        records.append({'model': name, 'seed': seed, **seconds,
                        'total_development_seconds': sum(seconds.values())})

costs = pd.DataFrame(records)
costs.to_csv(base / 'model_development_costs_by_seed.csv', index=False)
r.summarize_trials(base / 'model_development_costs_by_seed.csv',
    ['model'], ['tuning_seconds', 'shap_seconds', 'optimise_seconds',
                'final_fit_seconds', 'total_development_seconds']).to_csv(
    base / 'model_development_costs_summary.csv', index=False)
print(per_fit)
print(costs.groupby('model').mean(numeric_only=True))
