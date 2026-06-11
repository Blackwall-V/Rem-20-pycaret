# %% Reentrenamiento rápido para sincronizar columnas
import os
import sys
import pandas as pd
from pycaret.regression import setup, compare_models, finalize_model, save_model

project_root = "/home/v/Projects/Rem-20-pycaret"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.components.data_preprocessing import DataPreprocessing

# 1. Cargar y limpiar datos con tu lógica actual
RAW_DATA_PATH = "/home/v/Projects/Rem-20-pycaret/data/raw/DS4-21-Telco-Customer-Churn.csv"
preprocessor = DataPreprocessing(raw_data_path=RAW_DATA_PATH)
clean_df = preprocessor.clean_data().reset_index(drop=True)

print("Inicializando nuevo Setup...")
# 2. Inicializar setup real (dejamos que PyCaret maneje su split por defecto)
setup(data=clean_df, target="MonthlyCharges", verbose=False, html=False, session_id=123)

print("Entrenando modelo rápido (Fast Mode)...")
# 3. Entrenar un modelo rápido (ej. un árbol de decisión o una regresión lineal)
best_model = compare_models(budget_time=1, verbose=False) 
final_model = finalize_model(best_model)

print("Guardando nuevo pipeline sincronizado...")
# 4. Sobreescribimos el archivo pkl viejo
MODEL_OUTPUT_PATH = "/home/v/Projects/Rem-20-pycaret/storage/models/best_regression_pipeline"
save_model(final_model, MODEL_OUTPUT_PATH)
print("¡Pipeline sincronizado con éxito!")


# %% Visualización del modelo sincronizado
from pycaret.regression import load_model, plot_model

MODEL_OUTPUT_PATH = "/home/v/Projects/Rem-20-pycaret/storage/models/best_regression_pipeline"
pipeline_cargado = load_model(MODEL_OUTPUT_PATH)

print("\n--- Generando Gráficos ---")
plot_model(pipeline_cargado, plot="feature", save=True)
plot_model(pipeline_cargado, plot="error", save=True)
