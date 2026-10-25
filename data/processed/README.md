# Processed monitoring dataset

`All_with_group.csv` contains 24,000 task profiles in 800 consecutive blocks of 30 rows. The `group` field identifies positional blocks; it is not the original Hadoop job ID. Record order is retained for within-job label construction.

| Field | Description |
| --- | --- |
| `group` | Positional job-group identifier |
| `dataSplit`, `map`, `shuffle`, `reduce`, `dataCombine` | Recorded execution-stage times |
| `dataLocality`, `networkFault` | Context fields excluded from model inputs |
| `networkTraffic`, `CPUusage`, `Musage` | Network and resource-utilisation measurements |
| `makespan` | Sum of the five execution-stage times |
| `normalized_makespan` | Within-job normalised makespan |
| `performance` | Status label: 0 = degraded, 1 = normal |

The `makespan` and `normalized_makespan` columns are excluded from the predictor matrix, but the five constituent execution times remain model inputs. Measurement units are not documented in the CSV. Dataset-use terms are described in `../../DATA_LICENSE.md`.
