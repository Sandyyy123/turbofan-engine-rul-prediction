# Validation Report - Project 11: NASA Turbofan RUL

## Compact Summary

**Overall: PASS-WITH-WARNINGS**

The scaffold is structurally sound and reproducible. Notebook JSON parses, both Python scripts compile, the HTML deck is fully self-contained (zero external resources), em-dash count is zero across all artefacts, no AI-tell phrases were found, and the checkpoint JSON includes all required schema keys plus extras. The manuscript is 4060 words (within the 4000-5000 target band), IMRaD-complete, and all named methods (RandomForest, LSTM, 1D-Transformer, GroupShuffleSplit, piece-wise RUL cap=125, Adam, early stop, NASA score) are implemented in `src/`. CrossRef live verification of 5 random references all returned HTTP 200 with title matches. Two warnings: (1) inline citation `[Vaswani 2017]` has no entry in `reports/references.md` (orphan), and (2) Phase-1 scaffold has no executed-model artefacts in `deliverables/` other than the HTML deck (expected for a scaffold-only project; not a fail).

---

## Task-by-task findings

### 1. Notebook validity
- [PASS] `notebooks/01_EDA.ipynb` parses as valid JSON (`json.load` succeeded).

### 2. Python script syntax
- [PASS] `src/model_baseline.py` parses cleanly (`ast.parse` OK).
- [PASS] `src/model_advanced.py` parses cleanly (`ast.parse` OK).

### 3. Manuscript word count
- [PASS] `manuscripts/manuscript.md` = **4060 words** (target 4000-5000, inside band).

### 4. Self-contained HTML
- [PASS] `deliverables/presentation.html` has **0 external `href="http`/`src="http`** matches. Inline-only confirmed.

### 5. IMRaD completeness
- [PASS] All required sections present: Title (H1), Abstract, 1. Introduction, 2. Methods (with 9 sub-sections including Dataset, Preprocessing, Train/val split, Baseline, Advanced, Evaluation metrics, Reproducibility, Computational footprint, Hyperparameter choices), 3. Results, 4. Discussion, 5. Conclusion, References.

### 6. Method drift (manuscript Methods vs `src/`)
- [PASS] RandomForestRegressor (Methods 2.4) -> present in `model_baseline.py` (`from sklearn.ensemble import RandomForestRegressor`).
- [PASS] Hand-crafted lag/window stats - mean, std, min, max, slope (Methods 2.4) -> `_window_stats` in `model_baseline.py`.
- [PASS] Piece-wise linear RUL with cap=125 (Methods 2.2) -> `add_piecewise_rul(cap=RUL_CAP)` in both scripts; `RUL_CAP = 125`.
- [PASS] Drop low-variance sensors with std < 1e-3 (Methods 2.2) -> `keep_sensors`/`drop` logic in both scripts.
- [PASS] z-score on training-set statistics (Methods 2.2) -> standardisation block in both scripts.
- [PASS] GroupShuffleSplit by engine, 20% holdout, seed 42 (Methods 2.3) -> `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)` in both scripts.
- [PASS] LSTM 2-layer (Methods 2.5) -> `LSTMRegressor` with `nn.LSTM` in `model_advanced.py`.
- [PASS] 1D-Transformer with sinusoidal positional encoding (Methods 2.5) -> `PositionalEncoding` + `TransformerRegressor` (`nn.TransformerEncoder`) in `model_advanced.py`.
- [PASS] Adam, lr 1e-3, MSE, early stop on val RMSE (Methods 2.5) -> `torch.optim.Adam`, `LR=1e-3`, MSE loss, early-stop block in `model_advanced.py`.
- [PASS] RMSE, MAE metrics (Methods 2.6) -> `mean_squared_error`, `mean_absolute_error` in both scripts.
- [PASS] NASA asymmetric scoring (Methods 2.6) -> implemented in both scripts (function over `d = y_pred - y_true`).
- [PASS] Sliding window length 30 (Methods 2.5) -> `WINDOW = 30` in both scripts.
- No method drift detected.

### 7. Citation drift (manuscript inline cites vs `reports/references.md`)
- 27 unique citation keys extracted. 26 map cleanly to entries in `references.md`.
- [WARN] **`Vaswani 2017`** (Methods 2.5, sinusoidal positional encoding for the 1D-Transformer) is **orphan** - no corresponding entry in `references.md`. The manuscript flags it parenthetically as "(cited in survey reviews)", but a direct entry should be added (DOI suggestion: `10.48550/arXiv.1706.03762`, "Attention Is All You Need").
- [PASS] All other inline keys (Saxena 2008a/b, Heimes 2008, Zheng 2017, Hochreiter 1997, Greff 2017, Breiman 2001, Wu 2018, Li 2018, Listou Ellefsen 2019, Ma 2021, Xiang 2021, Qin 2021, Zhao 2020, Chen 2025, Wu 2024, Singh 2025, Wen 2021, Shao 2019, Guo 2019, Lei 2018, Carvalho 2019, Fink 2020, Cox 1972, Ishwaran 2008, Bektas 2018, Li 2021) resolve to a numbered entry in `references.md`.

### 8. CrossRef live re-verification of 5 random references
All 5 randomly selected DOIs returned HTTP 200 with title-string match against the local entry:
- [PASS] `10.1023/A:1010933404324` -> "Random Forests" (Breiman 2001). Status 200.
- [PASS] `10.1016/j.ymssp.2017.11.016` -> "Machinery health prognostics: A systematic review from data acquisition to RUL prediction" (Lei 2018). Status 200.
- [PASS] `10.1016/j.engappai.2020.103678` -> "Potential, challenges and future directions for deep learning in prognostics and health management applications" (Fink 2020). Status 200.
- [PASS] `10.1109/ICPHM.2017.7998311` -> "Long Short-Term Memory Network for Remaining Useful Life estimation" (Zheng 2017). Status 200.
- [PASS] `10.1162/neco.1997.9.8.1735` -> "Long Short-Term Memory" (Hochreiter 1997). Status 200.

### 9. Em-dash scan
- [PASS] **0 em-dash characters** across `brief.md`, `notebooks/01_EDA.ipynb`, `reports/references.md`, `src/model_baseline.py`, `src/model_advanced.py`, `manuscripts/manuscript.md`, `deliverables/presentation.html`.

### 10. AI-tell scan
- [PASS] No occurrences of `verified by N agents`, `AI-verified`, or `cross-checked by Claude` anywhere in the project tree (recursive grep returned no hits).

### 11. Checkpoint schema
- [PASS] `checkpoint.json` keys: `['project_number', 'title', 'methodology', 'phase', 'status', 'needs_main_session_execution', 'blockers']`. All four required fields (`project_number`, `title`, `methodology`, `status`) are present. No missing fields.

### Bonus: Saved-model artefacts in `deliverables/`
- [WARN] `deliverables/` contains only `presentation.html`. No `.pkl`, `.pt`, or `.png` artefacts.
- Project 11 is **scaffold-only** (Phase 1, scripts ready-to-run but not executed). Per QA rules this is a WARN not a FAIL, and the brief explicitly states Phase 2 will execute the scripts and produce the saved model files.

---

## Recommended fixes (low-effort)

1. Add a `Vaswani 2017` entry to `reports/references.md` under "Foundational machine learning and statistics" (or a new "Transformer" subsection) with DOI `10.48550/arXiv.1706.03762`. This closes the only citation-drift warning.
2. Optional: drop the parenthetical "(cited in survey reviews)" qualifier in Methods 2.5 once the Vaswani entry is added.

No other changes needed for Phase 1 sign-off.
