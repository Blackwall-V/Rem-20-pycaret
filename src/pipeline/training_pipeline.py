import os
import sys

# 1. FIX PATH FIRST: Calculate and inject project root before doing ANY custom imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. NOW DO IMPORTS: Python can now safely see the 'src' directory
import pandas as pd
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer

def run_pipeline():
    RAW_DATA_PATH = "/home/v/Projects/Rem-20-pycaret/data/raw/indicadores_rem20_20260325.csv"  
    MODEL_OUTPUT_PATH = "storage/models/best_regression_pipeline"
    TARGET_COL = "INDICE_OCUPACIONAL"                  

    # guard check
    if not os.path.exists(RAW_DATA_PATH):
        print(f"[Error] Raw data file not found at {RAW_DATA_PATH}. Please place your data in the repo.")
        return

    # Step 1: Execute custom data preprocessing
    print("Executing custom preprocessing module...")
    preprocessor = DataPreprocessing(raw_data_path=RAW_DATA_PATH)
    clean_df = preprocessor.clean_data()
    
    # Save the processed data step into the repo for lineage tracking
    clean_df.to_csv("data/processed/cleaned_dataset.csv", index=False)
    
    # Step 2: Initialize and execute PyCaret AutoML Regression
    trainer = ModelTrainer(target_column=TARGET_COL, session_id=123)
    
    # Metrics options to sort by: 'R2', 'RMSE', 'MAE', 'MSE', 'MAPE'
    trainer.initiate_training(
        train_df=clean_df, 
        model_save_path=MODEL_OUTPUT_PATH, 
        optimize_metric="R2" 
    )

if __name__ == "__main__":
    run_pipeline()
