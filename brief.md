# Project 11: NASA Turbofan Engine RUL Prediction

## One-liner
Predict the Remaining Useful Life (RUL) of aircraft turbofan engines from multivariate sensor sequences using a Random Forest baseline and a deep sequence model (LSTM / 1D-Transformer).

## Why this matters
Predictive maintenance is one of the highest-value AI use cases in DACH industrial settings (Siemens, MTU Aero Engines, Lufthansa Technik, ZF, Bosch). Replacing fixed-interval maintenance with condition-based RUL prediction reduces unscheduled downtime, lowers spare-part inventories, and extends component life. The NASA C-MAPSS turbofan dataset is the de-facto academic benchmark for this problem and serves as the twin of project #07 (industrial anomaly detection): same domain, different framing (regression on time-to-failure rather than binary anomaly score).

## Methodology

- **Task**: regression of remaining cycles to failure on multivariate sensor sequences.
- **Dataset**: NASA C-MAPSS turbofan engine simulation (Saxena et al., 2008) - subsets FD001 to FD004. Primary subset: FD001 (single operating condition, single fault mode, 100 train units, 100 test units, 21 sensors + 3 operational settings).
- **RUL labels**: piece-wise linear with cap at 125 cycles (literature standard, see Heimes 2008, Zheng 2017).
- **Baseline** (`src/model_baseline.py`): Random Forest regressor on hand-crafted lag/window statistics (mean, std, slope of last 30 cycles per sensor).
- **Advanced** (`src/model_advanced.py`): LSTM / 1D-Transformer on raw sensor sequences (window length 30, stride 1).
- **Metrics**: RMSE, MAE, NASA scoring function (asymmetric penalty - over-estimation of RUL is penalised harder than under-estimation).

## Domain context
- C-MAPSS is a NASA-published thermodynamic simulation of a 90,000-lb thrust commercial turbofan.
- 4 sub-datasets vary by number of operating conditions and fault modes.
- Each engine starts healthy and degrades stochastically until failure.
- Train set contains complete run-to-failure trajectories; test set is truncated at a random earlier cycle and the goal is to predict remaining cycles.

## Phase 1 deliverables
- Folder structure mirrored from `liora_projects/05_co2_emissions/`
- `data/README.md` with Kaggle CLI command and dataset notes
- `notebooks/01_EDA.ipynb` (raw, not executed)
- `reports/references.md` with 20+ verified references (CrossRef + Europe PMC)
- `src/model_baseline.py` and `src/model_advanced.py` (both ready-to-run, not executed)
- `manuscripts/manuscript.md` (4000-5000 words, IMRaD)
- `deliverables/presentation.html` (8-12 sections, inline CSS, offline-capable)
- `checkpoint.json` (status JSON)

## Phase 2 (later, in main session)
- Execute notebooks and scripts on the actual dataset
- Fill in numeric results, plots, and tables
- Re-run manuscript build script to inject canonical numbers from CSVs
- Push to GitHub
