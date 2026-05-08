# Improvements - Project 11: NASA Turbofan RUL Prediction

**Reviewer role**: IMPROVER (Role B)
**Date**: 2026-05-08
**Scope**: Phase 1 scaffold (code-only, not yet executed). Recommendations only, no file edits.

---

## Top recommendation

**Replace the LSTM-only deep model with a CNN-LSTM hybrid (or attention-augmented BiLSTM) and run all four C-MAPSS subsets, not just FD001.** The current advanced script defaults to a vanilla 2-layer LSTM at hidden=64, which the literature (Ma 2021, Listou Ellefsen 2019, Wu 2024 survey) shows is consistently 1-3 RMSE cycles behind a CNN-LSTM or BiLSTM-with-attention on FD001 and 5-8 cycles behind on FD002/FD004. A single architectural change (1D-conv front-end before the LSTM) and a multi-subset evaluation loop would move the project from "competent baseline" to "publication-credible benchmark" with no extra data and roughly 30 minutes of additional GPU time per subset. This is the single highest-leverage change: it strengthens the manuscript Results section, makes the cross-subset generalisation story testable, and aligns the deliverable with what a DACH industrial reviewer (MTU, Lufthansa Technik) would actually expect to see.

---

## Weaknesses and proposed improvements

### 1. Multi-subset evaluation missing (HIGH)

**Weakness**: Both `model_baseline.py` and `model_advanced.py` hard-code `SUBSET = "FD001"`. The brief explicitly mentions FD001-FD004 and the manuscript discusses domain transfer across subsets, but no code path actually loops over them. FD002 and FD004 (six operating conditions, two fault modes for FD004) are where the methodology is genuinely tested, and any reviewer who knows the dataset will notice the omission immediately.

**Action**: Lift `SUBSET` to a CLI flag (`argparse` or env var) and add a thin `run_all_subsets.py` driver that calls each script for FD001-FD004 in turn, writing per-subset metric JSONs. Add per-regime z-score normalisation for FD002/FD004 (cluster the three op_settings into 6 regimes via KMeans, then standardise sensors within each regime, as in Listou Ellefsen 2019 section 3.2). The manuscript's Results table can then have four rows instead of one, which is the standard format in the literature.

### 2. Deep model is one architectural generation behind (HIGH)

**Weakness**: A 2-layer LSTM with hidden=64 is the 2017 Zheng et al. baseline. The current C-MAPSS state-of-the-art on FD001 is roughly RMSE 11.5-12.5 cycles (CNN-LSTM hybrids, BiLSTM with attention, or the GDAU model from Qin 2021 already in the references). Vanilla LSTMs sit closer to RMSE 13.5-14.5. The Transformer alternative is a fairer comparison but its parameter budget (d_model=64, 2 layers) is too small to outperform the LSTM and too large to be efficient.

**Action**: Add a third architecture, `CNNLSTM` (Conv1d(filters=32, kernel=5) -> Conv1d(64, 3) -> LSTM(64, 2 layers) -> linear head). Cite Ma 2021 (`10.1109/TII.2020.2991796`, already in references) as justification. Keep the LSTM and Transformer for ablation but report CNN-LSTM as the headline number. Expected gain on FD001: 1-2 cycles RMSE.

### 3. No uncertainty quantification (HIGH)

**Weakness**: The pipeline emits a single point prediction per engine. Maintenance decisions in aviation are risk-weighted: a maintenance manager needs P(RUL < 20 cycles) more than they need a point estimate. Section 4.3 of the manuscript flags this as a limitation but does not act on it. Li 2021 (already in references) is cited but not implemented.

**Action**: Add MC-Dropout inference at test time (run the LSTM 50 times with dropout enabled, report the mean and 5/95 percentiles per engine). This is a 30-line addition to `predict()` in `model_advanced.py` and turns the deliverable from "RMSE on a benchmark" into "calibrated risk estimates with prediction intervals", which is the language a DACH industrial client actually wants.

### 4. NASA scoring asymmetry not exploited in training (HIGH)

**Weakness**: Both models train with symmetric MSE loss but are evaluated with the asymmetric NASA score. This means the training objective is misaligned with the deployment metric. Recent work (Wu 2024 survey, section 4.2) notes that asymmetric losses (e.g. quantile loss at quantile 0.4-0.45, or a custom NASA-aware loss) reduce the NASA score by 15-25 percent on FD001 with no architectural change.

**Action**: Add an asymmetric loss option to `model_advanced.py`: `loss_fn = AsymmetricMSE(over_weight=1.5, under_weight=1.0)` where over-predictions are penalised 50 percent harder. Report both standard-MSE and asymmetric-MSE results in the Results table to demonstrate the trade-off.

### 5. No cross-validation, single seed, single split (MEDIUM)

**Weakness**: `GroupShuffleSplit(n_splits=1, random_state=42)` produces one train/val split. A single-split number with no confidence interval is fragile. The cohort's anti-fabrication rules implicitly require traceable numbers, but without CV the headline RMSE has no error bar and the comparison between models can be a coin flip.

**Action**: Switch to `GroupKFold(n_splits=5)` and report mean +/- std RMSE across folds for both baseline and advanced. Run each fold with three seeds (42, 1337, 2026) for the deep model, total 15 runs per subset. This adds 1-2 GPU-hours but removes the "is the difference between LSTM and Transformer real or noise?" question entirely. It also produces error bars for the manuscript Results section, which is currently a row of point estimates.

### 6. Hand-crafted feature set is shallow for the baseline (MEDIUM)

**Weakness**: The baseline computes mean/std/min/max/slope per sensor over a 30-cycle window. This misses two well-documented feature families: (a) frequency-domain features (FFT band powers, dominant frequency) that capture vibration signatures of rotating-machinery degradation, and (b) trend-removal features (e.g. detrended slope, second derivative) that distinguish steady drift from regime changes. Tsfresh and Catch22 toolkits compute these in seconds.

**Action**: Add two feature blocks to `build_window_features`: (i) FFT band-power in 3-5 frequency bins per sensor, (ii) Catch22 (`pip install catch22`) over the 30-cycle window. This pushes the baseline feature count from approximately 85 to approximately 200 per cycle. Random Forest handles the dimensionality fine; expected RMSE improvement on FD001: 0.5-1.0 cycles based on prior reports for similar feature additions.

### 7. No requirements.txt, no random_seed in notebook, no data versioning (MEDIUM)

**Weakness**: The project folder has no `requirements.txt` or `pyproject.toml`. Reproducibility hinges on a reader installing the right versions of `scikit-learn`, `torch`, `numpy`, `pandas`, `joblib`, `matplotlib`, `seaborn`. The EDA notebook does not seed numpy and does not pin a Kaggle dataset version. The brief notes ~25 MB but does not record an md5/sha256 hash of the canonical zip.

**Action**: Add a `requirements.txt` with pinned versions (the ones the notebook will be developed against): `scikit-learn==1.5.*, torch==2.4.*, numpy==2.0.*, pandas==2.2.*, matplotlib==3.9.*, seaborn==0.13.*, joblib==1.4.*, jupyter==1.1.*`. Add a `data/CHECKSUMS.txt` with sha256 of `nasa-cmaps.zip` and each `train_FDxxx.txt` after first download. Add `np.random.seed(42)` in the EDA notebook's first code cell.

### 8. Manuscript prose is thorough but reproducibility hooks are weak (MEDIUM)

**Weakness**: The manuscript word count target (4000-5000) is met (4060), but every numerical claim is currently `<TBD after model run>` and the Results section is a stub. More importantly, the manuscript references a `build script` that injects canonical numbers from JSON at finalisation time but no such script exists in `src/` (no `manuscripts/build_manuscript.py` or similar). This is the exact failure mode the project's own anti-hallucination rules are designed to prevent.

**Action**: Add `manuscripts/build_manuscript.py` that reads `deliverables/metrics_baseline.json` and `deliverables/metrics_advanced_*.json`, replaces every `<TBD after model run>` token in a `manuscript_template.md` with the canonical number, and writes `manuscript_final.md`. This makes Phase 2 "fill in numbers" a single command rather than a manual edit, and structurally prevents the kind of stale-number drift the cohort's `feedback_no_hallucination_no_lying` rule was written to address.

### 9. Presentation HTML lacks a results-comparison chart placeholder (LOW)

**Weakness**: The presentation has 8-12 sections covering motivation, data, methods, etc., but no visual placeholder for the eventual results table or per-engine residual plot. A client-facing Phase 2 reader will see code and prose but no chart, which weakens the business-audience clarity the deliverables are supposed to optimise for.

**Action**: Add two chart sections to `presentation.html`: (i) a results comparison table (RF / LSTM / Transformer / CNN-LSTM, with RMSE/MAE/NASA-score columns), (ii) an embedded base64-encoded `residuals.png` placeholder generated by a one-liner in the model scripts (`plt.scatter(y_true, y_pred); plt.savefig('deliverables/residuals.png')`). Both are inline-only, no external resources, consistent with the offline-capable rule.

### 10. Test-set evaluation discards calibration information (LOW)

**Weakness**: Test-set scoring uses only the final cycle of each test engine (matching the NASA challenge protocol). This is correct for benchmark comparability, but it discards information from earlier cycles that could measure calibration over the degradation trajectory. A reliability engineer wants to know whether the model's RUL estimate is stable over the last 50 cycles or jumps around.

**Action**: Add a secondary evaluation that computes RMSE at each remaining-cycle bin (e.g. RMSE when true RUL is 100-125 vs 50-100 vs 0-50). Tabulate as a "calibration over degradation" plot. This is a 20-line addition to `model_advanced.py` and gives the Discussion section a concrete result rather than a future-tense promise.

---

## Priority summary

| # | Improvement | Priority |
|---|-------------|----------|
| 1 | Multi-subset evaluation (FD001-FD004) | HIGH |
| 2 | CNN-LSTM hybrid as headline architecture | HIGH |
| 3 | MC-Dropout uncertainty intervals | HIGH |
| 4 | Asymmetric loss aligned to NASA score | HIGH |
| 5 | GroupKFold CV with multi-seed | MEDIUM |
| 6 | FFT and Catch22 features for baseline | MEDIUM |
| 7 | requirements.txt, seeds, data checksums | MEDIUM |
| 8 | Manuscript build script (numeric injection) | MEDIUM |
| 9 | Results chart placeholders in presentation | LOW |
| 10 | Calibration-over-degradation evaluation | LOW |

---

## What is already strong (do not change)

- Correct piece-wise linear RUL with cap=125 (Heimes 2008, Zheng 2017 standard).
- Per-engine train/val split via `GroupShuffleSplit` (avoids within-engine leakage; this is one of the most common errors in C-MAPSS work).
- Three-metric reporting (RMSE, MAE, NASA score) covers magnitude and asymmetric cost.
- 33-reference verified bibliography with CrossRef/Europe PMC verification, DOIs/PMIDs only.
- Self-contained HTML presentation with no external resources.
- Manuscript transparently flags `<TBD after model run>` placeholders rather than fabricating numbers.
- Brief and data README cleanly separate Phase 1 (scaffold) from Phase 2 (execution).

---

**File**: `/root/AI/liora_projects/11_nasa_turbofan/improvements.md`
**Status**: Role B (IMPROVER) complete.
