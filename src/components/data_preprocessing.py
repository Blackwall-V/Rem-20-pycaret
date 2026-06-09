import pandas as pd

class DataPreprocessing:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path

    def clean_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.raw_data_path)

        return df
