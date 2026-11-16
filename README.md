# Leave-Unit-Out Benchmarking of Early-Warning Models for Geothermal Turbines Using Physics-Based Synthetic Sensor Data

Supplementary code for the manuscript. It reproduces the leave-unit-out benchmark of logistic regression, random forest and XGBoost early-warning models on a physics-based synthetic sensor dataset of geothermal steam turbines.

## Contents

| Path | Purpose |
|---|---|
| `src/sim.py` | Physics-based sensor simulator (IAPWS-IF97 steam properties, three degradation modes, causal feature extraction). |
| `src/exp.py` | Main experiment: leave-unit-out CV, LR/RF/XGB with none / class-weight / SMOTE / ADASYN, threshold chosen on validation units, random-split and shift tests, permutation importance. |
| `src/extra.py` | Additional analyses: feature ablation, hyperparameter sensitivity, and a matched-size random row split. |
| `src/analyze.py` | Aggregates per-replicate outputs into tables and figures. |
| `src/aggregate_extra.py` | Holm correction over paired tests and aggregation of the additional analyses. |
| `results/` | Small summary tables produced by the scripts, for comparison with your own run. |

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9 or newer is recommended.

## Reproduction

Run from the `src/` directory. Each replicate takes roughly 13–17 minutes on one CPU core. The three replicates (0, 1, 2) are independent and can run in parallel.

```bash
cd src

# 1. Main experiment, three replicates
python3 exp.py 0 rep0
python3 exp.py 1 rep1
python3 exp.py 2 rep2

# 2. Additional analyses, three replicates
python3 extra.py 0
python3 extra.py 1
python3 extra.py 2

# 3. Aggregate main results, tables and figures
python3 analyze.py

# 4. Holm correction and additional-analysis tables
python3 aggregate_extra.py
```

Step 1 writes `rep{R}_metrics.csv`, `rep{R}_preds.parquet` and, for replicate 0, `rep0_data.parquet`. Step 2 writes `extra_rep{R}.csv`. Step 3 needs the outputs of step 1 and writes `table_*.csv`, `tests.json`, and `fig/`. Step 4 needs step 2 and `tests.json` from step 3.

Random seeds: simulated fleets use seeds 1000–1002, shift fleets use 5000–5002.

## Notes

- `exp.py` and `extra.py` each recompute the simulated dataset from the seed, so no pre-generated data file is needed.
- Output files are not included in this repository. They are reproduced by the commands above.
- Summary tables in `results/` come from the original run and serve as a reference.
