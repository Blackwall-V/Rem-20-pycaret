import pandas as pd

class DataPreprocessing:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path

    def clean_data(self) -> pd.DataFrame:
        try:
            # 1. First attempt standard reading
            df = pd.read_csv(self.raw_data_path)
            
            # If it read it as a single giant column, it might be a semicolon CSV
            if df.shape[1] == 1:
                df = pd.read_csv(self.raw_data_path, sep=';')
                
        except pd.errors.ParserError:
            print("[Warning] Standard parsing failed. Re-trying with on_bad_lines='skip'...")
            # 2. Resilient fallback: Skips corrupted/footer rows instead of crashing
            df = pd.read_csv(self.raw_data_path, sep=None, engine='python', on_bad_lines='skip')
            
        # Clean white space from column headers if any exist
        df.columns = df.columns.str.strip()
   
        return df
