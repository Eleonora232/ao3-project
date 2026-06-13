import os
import ast
from pathlib import Path
import pandas as pd

class AO3DatasetLoader:
    """
    A class to handle Kaggle setup, authentication, file download, loading,
    and preprocessing of the AO3 Fanworks dataset.
    """
    def __init__(self, dataset_ref='csutliff1/ao3-dataset-collected-4-oct-2025', filename='random works - Oct 2025.csv', download_path='.'):
        self.dataset_ref = dataset_ref
        self.filename = filename
        self.download_path = Path(download_path)
        self.target_filepath = self.download_path / self.filename
        self.encoded_filename = 'random%20works%20-%20Oct%202025.csv'
        self.username = 'eleonora232'
        self.fallback_token = 'KGAT_27cc5a6e6bb9e3ee5304483072e6b5d8'

    def setup_kaggle_credentials(self):
        """Sets up Kaggle API credentials from ~/.kaggle/access_token or fallback."""
        os.environ['KAGGLE_USERNAME'] = self.username
        
        token_path = Path.home() / '.kaggle' / 'access_token'
        if token_path.exists():
            os.environ['KAGGLE_KEY'] = token_path.read_text().strip()
            print("Kaggle credentials set from ~/.kaggle/access_token.")
        else:
            os.environ['KAGGLE_KEY'] = self.fallback_token
            print("Kaggle credentials set using fallback token.")

    def download_dataset(self):
        """Downloads the dataset file if it does not already exist."""
        if not self.target_filepath.exists():
            self.setup_kaggle_credentials()
            
            # Dynamically import to avoid early initialization crashes during top-level imports
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            print(f"Initializing Kaggle API and downloading '{self.filename}'...")
            api = KaggleApi()
            api.authenticate()
            
            api.dataset_download_file(self.dataset_ref, self.filename, path=str(self.download_path))
            
            # Check for URL-encoded name and rename it
            encoded_filepath = self.download_path / self.encoded_filename
            if encoded_filepath.exists():
                encoded_filepath.rename(self.target_filepath)
                print(f"Downloaded and renamed to: {self.target_filepath}")
            else:
                print(f"Downloaded file directly as: {self.target_filepath}")
        else:
            print(f"File already exists at: {self.target_filepath}. Skipping download.")

    def load_data(self) -> pd.DataFrame:
        """Loads the CSV file into a pandas DataFrame."""
        if not self.target_filepath.exists():
            raise FileNotFoundError(f"Dataset file not found at {self.target_filepath}. Please run download_dataset() first.")
        print(f"Loading dataset from {self.target_filepath}...")
        return pd.read_csv(self.target_filepath)

    @staticmethod
    def parse_list(val):
        """Helper to parse columns stored as string representations of Python lists."""
        if pd.isna(val):
            return []
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                # Fallback for ill-formed list strings
                return [item.strip() for item in val.strip('[]').replace("'", "").split(',') if item.strip()]
        return []

    def get_preprocessed_dataframe(self) -> pd.DataFrame:
        """Downloads (if necessary), loads, pre-processes, and returns the DataFrame."""
        self.download_dataset()
        df = self.load_data()
        
        print("Parsing list columns (Authors, Fandom Tags, Freeform Tags)...")
        df['parsed_authors'] = df['Authors'].apply(self.parse_list)
        df['parsed_fandoms'] = df['Fandom Tags'].apply(self.parse_list)
        df['parsed_freeform'] = df['Freeform Tags'].apply(self.parse_list)
        print("Preprocessing complete.")
        
        return df
