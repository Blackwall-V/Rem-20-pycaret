# %% [markdown]
# # Rem-20 Pipeline Interactive Scratchpad

# %%
import os
import sys
import pandas as pd

# Fix path context so local modules load flawlessly
project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer

# %% [markdown]
# ### Step 1: Run Preprocessor and Inspect Data Frame Shape

# %%
RAW_DATA_PATH = "data/raw/indicadores_rem20_20260325.csv"
preprocessor = DataPreprocessing(raw_data_path=RAW_DATA_PATH)
clean_df = preprocessor.clean_data()

# Print directly inline to inspect your data
print(f"Dataframe loaded! Rows: {clean_df.shape[0]}, Columns: {clean_df.shape[1]}")
print(clean_df.head())

# %% [markdown]
# ### Step 2: Fast Training Block (MLflow Tracking turned OFF)

# %%
MODEL_OUTPUT_PATH = "storage/models/best_regression_pipeline"
TARGET_COL = "INDICE_OCUPACIONAL"

trainer = ModelTrainer(target_column=TARGET_COL, session_id=123)

# Note: track_experiment=False bypasses MLflow to optimize execution time!
trainer.initiate_training(
    train_df=clean_df, 
    model_save_path=MODEL_OUTPUT_PATH, 
    optimize_metric="R2",
    track_experiment=False  
)
