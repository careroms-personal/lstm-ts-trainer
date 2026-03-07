import pandas as pd

from models.config_models import LSTMTsTrainingConfig

from pathlib import Path

class Normalizer:
  def __init__(self, training_config: LSTMTsTrainingConfig):
    self.training_config = training_config

  def _normalize_data(self) -> pd.DataFrame:
    if self.training_config.dataset_config.dataset_files[0] == "*":
      file_list = list(Path(self.training_config.dataset_config.dataset_dir).glob("*.csv"))
    else:
      file_list = [
        Path(self.training_config.dataset_config.dataset_dir) / f
        for f in self.training_config.dataset_config.dataset_files
      ]

    if not file_list:
      raise ValueError(f"No CSV files found in {self.training_config.dataset_config.dataset_dir}")
    
    dfs = []

    timestamp_col = self.training_config.dataset_config.timestamp_col
    value_col = self.training_config.dataset_config.value_col

    for file in file_list:
      df = pd.read_csv(file)

      extracted_df = df[[timestamp_col, value_col]].copy()
      dfs.append(extracted_df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values(timestamp_col).reset_index(drop=True)

    combined_df[value_col] = combined_df[value_col] / self.training_config.prediction_boundary.ceiling_value
    
    return combined_df
    
  def execute(self) -> pd.DataFrame:
    return self._normalize_data()