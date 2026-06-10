import pandas as pd
from pycaret.regression import RegressionExperiment

class ModelTrainer:
    def __init__(self, target_column: str, session_id: int = 42):
        """
        Initializes the ModelTrainer component for Regression.
        """
        self.target = target_column
        self.session_id = session_id
        self.exp = RegressionExperiment()

    def initiate_training(self, train_df: pd.DataFrame, model_save_path: str, optimize_metric: str = "R2"):
        """
        Runs PyCaret setup, compares regression models, and saves the top performer.
        """
        print(f"--- Initializing PyCaret Regression Setup (Target: {self.target}) ---")
        
        #initialize regression enviroment
        self.exp.setup(
            data=train_df, 
            target=self.target, 
            session_id=self.session_id, 
            verbose=False,
            log_experiment='mlflow',
            experiment_name='rem20_regression'
        )
        
        print(f"--- Comparing Regression Models (Optimizing for: {optimize_metric}) ---")
        # compare models based on performance
        best_model = self.exp.compare_models(sort=optimize_metric)
        
        print(f"--- Finalizing and Saving the Best Model ---")
        # finalize model
        final_model = self.exp.finalize_model(best_model)

        #save the pipeline and transformation
        self.exp.save_model(final_model, model_save_path)
        print(f"Successfully saved production regression pipeline to: {model_save_path}.pkl")
        
        return final_model
