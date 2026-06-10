# Rem-20-pycaret

A professional, modular machine learning pipeline built with **PyCaret** to handle data preprocessing, automated regression modeling, evaluation, and pipeline serialization.

## 📁 Repository Structure

```text
Rem-20-pycaret/
├── config/                  # Configuration parameters
├── data/
│   ├── raw/                 # Place your immutable source CSVs here
│   └── processed/           # Automatically generated cleaned data
├── src/                     # Core source modules
│   ├── components/          # Ingestion, Preprocessing, and Training blocks
│   └── pipeline/            # End-to-end execution entry points
└── storage/                 # Generated artifacts (Logs and Saved Models)

<BS>🚀 Getting Started

Follow these steps to set up the environment and run the regression training pipeline locally.
1. Prerequisites

Ensure you have a stable version of Python 3.11 installed on your system.
2. Setup Environment & Install Dependencies

Run the following commands from the project root folder:

# Create a localized Python 3.11 virtual environment
python3.11 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade packaging tools and install requirements
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

<BS>3. Place Your Dataset

Drop your raw data file (.csv) into the target directory:
data/raw/indicadores_rem20_20260325.csv
⚡ Execution

Whenever you want to run the automated machine learning engine, activate your environment and execute the master pipeline script from the project root directory:

# 1. Activate environment (if not already active)
source .venv/bin/activate

# 2. Run the end-to-end regression pipeline
python src/pipeline/training_pipeline.py
