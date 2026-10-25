# Interpretable Infrastructure-level Workload Classification

Python implementation of the experiments in *Interpretable Infrastructure-level Workload Prediction for Large-scale Healthcare Analytics*.

The models classify duration-defined workload status from Hadoop task-level monitoring records using XGBoost, SVM and SHAP-informed CMA-ES feature weighting. SHAP, LIME and partial dependence provide model explanations. The classification labels depend on recorded execution-time components; the evaluation does not measure prospective detection of independently labelled degradation.

## Environment

The experimental implementation was developed in a Linux environment on Amazon Web Services (AWS). Hadoop workload execution and SmartMonit data collection used AWS infrastructure. Model training and evaluation were performed on a local workstation, as described in the manuscript.

Python 3.11 and the dependencies in `requirements.txt` are used for the experiments.

```bash
python -m pip install -r requirements.txt
python experiments/00_Environment_and_Data_Audit.py
```

## Data

`data/processed/All_with_group.csv` provides 24,000 monitoring profiles grouped into 800 consecutive blocks of 30 task profiles. The eight model inputs are `dataSplit`, `map`, `shuffle`, `reduce`, `dataCombine`, `networkTraffic`, `CPUusage` and `Musage`. The target `performance` uses `0` for degraded and `1` for normal duration-defined status. Aggregate makespan columns are excluded from the predictor matrix.

The original healthcare workload inputs (250K, 500K, 750K and 1M records) are not distributed. NEWS fidelity analysis uses local `data/news/NEWSsample.csv` and `data/news/NEWSgenerated.csv` inputs. Leave-one-scale-out evaluation requires `data/metadata/job_scale_map.csv`, which is not included in the processed monitoring table. See `DATA_ACCESS.md` and the accompanying data dictionaries.

## Python scripts

| Script | Analysis |
| --- | --- |
| `00_Environment_and_Data_Audit.py` | Data schema, integrity and label consistency |
| `01_Preprocessing_and_Label_Construction.py` | Predictors and duration-defined labels |
| `02_Outer_Splits.py` | Job-disjoint and profile-level partitions |
| `03_Hyperparameter_Tuning.py` | XGBoost and SVM model selection |
| `04_Main_Ablation_A0_A5.py` | A0–A5 ablation and random-search comparison |
| `05_Global_SHAP_and_LIME.py` | SHAP and LIME explanations |
| `06_Threshold_Sensitivity_A4.py` | Coefficients 1.25, 1.50, 1.75 and 2.00 |
| `07_Leave_One_Scale_Out.py` | Held-out workload-scale evaluation |
| `08_SHAP_Ranking_Stability.py` | SHAP ranking stability |
| `09_LIME_Stability_and_Fidelity.py` | Thirty-seed LIME analysis |
| `10_Partial_Dependence_Plots.py` | XGBoost and SVM partial dependence |
| `11_Runtime_Statistical_Tests_and_Tables.py` | Runtime summaries and paired statistical tests |
| `12_NEWS_Synthetic_Data_Fidelity.py` | Statistical fidelity of synthetic NEWS records |

## Experimental protocol

The main evaluation uses a fixed 80/20 job-disjoint partition (640/160 jobs), with model selection and optimisation performed within the training set. A0–A5 compare an unscaled baseline, standardisation, hyperparameter tuning, uniform CMA-ES initialisation, SHAP-informed CMA-ES initialisation, and matched-budget random feature-weight search. The additional threshold-sensitivity analysis uses a separate profile-level split. Each script saves its derived files under `results/` or `figures/`, which are excluded from version control.

Feature weights are normalised with softmax and applied as eight times the weight vector to standardised inputs. Optimisation uses degraded-class F1 as the fitness metric; classification results use macro-averaged scores. The five repetition seeds are 11, 23, 42, 67 and 89. The scripts use the full experimental settings by default. Shared methods are implemented in `src/repro.py`.

## Licence and citation

Software and documentation are available under the MIT `LICENSE`. The processed monitoring dataset is governed separately by `DATA_LICENSE.md`. Citation metadata are provided in `CITATION.cff`.
