# Rem-20-pycaret

A machine learning pipeline for regression tasks built with [PyCaret](https://pycaret.org/), leveraging automated model comparison, CatBoost, and MLflow experiment tracking.

---

## Overview

This project experiments with PyCaret's low-code AutoML capabilities to build and evaluate regression models. It uses CatBoost as a primary estimator (tracked via `catboost_info/`) and MLflow to log and compare experiment runs.

---

## Project Structure

```
Rem-20-pycaret/
├── catboost_info/        # Auto-generated CatBoost training logs (RMSE per iteration)
├── config/
│   └── config.yaml       # Project configuration (paths, parameters)
├── data/                 # Raw and processed datasets
├── src/                  # Source code (Python modules)
│   └── __init__.py
├── .gitignore
└── requirements.txt
```

> **Note:** Trained models are saved to `storage/models/` and experiment logs to `storage/logs/` (both excluded from version control via `.gitignore`). MLflow run data is stored in `mlruns/`.

---

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:

```
pycaret[full]
pandas
numpy
scikit-learn
```

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Blackwall-V/Rem-20-pycaret.git
cd Rem-20-pycaret
```

2. **Create and activate a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> `pycaret[full]` installs all optional extras including CatBoost, LightGBM, XGBoost, and more. This may take a few minutes.

---

## Usage

Place your dataset in the `data/` folder, then run the training pipeline from the `src/` directory. PyCaret's `setup()` will handle preprocessing, and `compare_models()` will benchmark multiple regressors automatically.

To view MLflow experiment results after a run:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Model Notes

- **Task type:** Regression
- **Primary metric:** RMSE
- **Best model:** CatBoost Regressor (based on training logs in `catboost_info/`)
- **Experiment tracking:** MLflow (`mlruns/`)

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| [PyCaret](https://pycaret.org/) | AutoML — model comparison, tuning, and pipeline |
| [CatBoost](https://catboost.ai/) | Gradient boosting regressor |
| [scikit-learn](https://scikit-learn.org/) | Base ML utilities |
| [pandas](https://pandas.pydata.org/) | Data manipulation |
| [numpy](https://numpy.org/) | Numerical computing |
| [MLflow](https://mlflow.org/) | Experiment tracking and logging |

---

## License

This project is open for experimentation.

---

## Author

**Blackwall-V** — [GitHub](https://github.com/Blackwall-V)
