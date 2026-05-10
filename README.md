![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Predictive Maintenance](https://img.shields.io/badge/task-RUL-orange) ![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)

# NASA Turbofan Engine Remaining Useful Life Prediction

Predicts Remaining Useful Life (RUL) of aircraft turbofan engines from multivariate sensor sequences using Random Forest and LSTM.

---

## Task

**Predictive Maintenance / RUL Regression**

---

## Architecture

```
Sensor Multivariate Time-series → Rolling Features → Random Forest / LSTM → RUL Regression
```

---

## Key Features

- RUL regression on 21 sensor channels across 4 degradation sub-datasets (FD001-FD004)
- Piecewise-linear RUL target (cap at 125 cycles)
- Random Forest baseline with rolling window aggregation features
- LSTM sequence model on raw sensor traces
- RMSE and score function evaluation (asymmetric penalty for late predictions)

---

## Dataset

[NASA CMAPSS Turbofan Degradation Dataset (Kaggle)](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Baseline model
│   └── model_advanced.py      # Advanced model
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup
├── reports/
│   └── references.md          # Verified references
├── deliverables/
│   └── presentation.html      # Self-contained HTML
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/turbofan-engine-rul-prediction.git
cd turbofan-engine-rul-prediction
pip install -r requirements.txt

# See data/README.md for dataset download
python src/model_baseline.py
python src/model_advanced.py
```

---

## Tech Stack

`PyTorch · scikit-learn · pandas · NumPy`

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Mössingen, Germany.

---

## License

MIT
