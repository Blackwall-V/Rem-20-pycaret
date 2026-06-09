from pycaret.classification import ClassificationExperiment 
import pandas as pd

class ModelTrainer:
    def __init__(self, target_column: str):
        self.target = target_column
        self.exp = ClassificationExperiment()

    def initiate_training(self, train_df: pd.DataFrame, model_save_path: str):
        # 1. Initialize 
        self.exp.setup(data=train_df, target=self.target, session_id=42, verbose=False)
        
        # 2. Compare 
        best_model = self.exp.compare_models()
        
        # 3. Save 
        self.exp.save_model(best_model, model_save_path)
        print(f"Model saved successfully to {model_save_path}")
