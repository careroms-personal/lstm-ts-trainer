import pandas as pd

from pathlib import Path

from models.config_models import LSTMTsTrainingConfig
from models.dataset_models import PreparedDataset

class DatasetPrepareExecutor:
  def __init__(self, lstm_training_config: LSTMTsTrainingConfig):
    self.dataset_config = lstm_training_config.dataset_config
    self.validation_config = lstm_training_config.dataset_config.validation_config
    self.prediction_boundary = lstm_training_config.prediction_boundary

  def _load_and_merge(self, dataset_dir: str, dataset_files: list[str]):
    if dataset_files[0] == "*":
      file_list = list(Path(dataset_dir).glob("*.csv"))
    else:
      file_list = [
        Path(dataset_dir) / f
        for f in dataset_files
      ]

    timestamp_col = self.dataset_config.timestamp_col
    value_col = self.dataset_config.value_col

    dfs = []

    for file in file_list:
      df = pd.read_csv(file)
      extracted_df = df[[timestamp_col, value_col]].copy()
      dfs.append(extracted_df)

    combined_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values(timestamp_col).reset_index(drop=True)

    return combined_df

  
  def _normalize(self, df: pd.DataFrame):
    new_df = df.copy()
    new_df[self.dataset_config.value_col] = new_df[self.dataset_config.value_col] / self.prediction_boundary.ceiling_value

    return new_df

  def execute(self):
    training_df = self._load_and_merge(
      self.dataset_config.training_data.dataset_dir, 
      self.dataset_config.training_data.dataset_files,
    )

    normalized_training_df = self._normalize(training_df)

    if self.validation_config.external_data.enabled:
      validation_df = self._load_and_merge(self.validation_config.external_data.dataset_dir, self.validation_config.external_data.dataset_files)

      normalized_validation_df = self._normalize(validation_df)
      split_amount = None
    else:
      normalized_validation_df = None
      split_amount = self.validation_config.internal_data.split_amount

    prepared_dataset = PreparedDataset(
      training_dataset=normalized_training_df,
      validation_dataset=normalized_validation_df,
      split_amount=split_amount,
      ceiling_value=self.prediction_boundary.ceiling_value,
      floor_value=self.prediction_boundary.floor_value,
      timestamp_col=self.dataset_config.timestamp_col,
      value_col=self.dataset_config.value_col,
    )

    return prepared_dataset