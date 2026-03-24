import pandas as pd
import numpy as np # type: ignore

from pathlib import Path

from models.config_models import LSTMTsTrainingConfig
from models.dataset_models import PreparedDataset, NPDataset
from sklearn.preprocessing import MinMaxScaler # type: ignore

class DatasetPrepareExecutor:
  def __init__(self, lstm_training_config: LSTMTsTrainingConfig):
    self.dataset_config = lstm_training_config.dataset_config
    self.validation_config = lstm_training_config.dataset_config.validation_config
    self.test_prediction_config = lstm_training_config.dataset_config.test_prediction_config
    self.windows_size = lstm_training_config.lstm_config.windows_size

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
  
  def _find_min_max_scale(self, df: pd.DataFrame):
    self.scaler = MinMaxScaler()
    self.scaler.fit(df[[self.dataset_config.value_col]])
  
  def _normalize(self, df: pd.DataFrame):
    new_df = df.copy()

    new_df[self.dataset_config.value_col] = self.scaler.transform(
      new_df[[self.dataset_config.value_col]]
    )

    return new_df
  
  def _df_to_np(self, df: pd.DataFrame) -> NPDataset:
    values = df[self.dataset_config.value_col].values
    
    X, y = [], []

    for i in range(len(values) - self.windows_size):
      X.append(values[i:i + self.windows_size])
      y.append(values[i + self.windows_size])

    return NPDataset(
      X=np.array(X).reshape(-1, self.windows_size, 1),
      y=np.array(y),
    )

  def execute(self):
    training_df = self._load_and_merge(
      self.dataset_config.training_data.dataset_dir, 
      self.dataset_config.training_data.dataset_files,
    )

    if self.validation_config.external_data.enabled:
      validation_df = self._load_and_merge(self.validation_config.external_data.dataset_dir, self.validation_config.external_data.dataset_files)     

      combined_df = pd.concat([training_df, validation_df])
      self._find_min_max_scale(combined_df)

      normalized_training_df = self._normalize(training_df)
      normalized_validation_df = self._normalize(validation_df)

      split_amount = None
    else:
      self._find_min_max_scale(training_df)
      
      normalized_training_df = self._normalize(training_df)
      normalized_validation_df = None

      split_amount = self.validation_config.internal_data.split_amount

    if self.test_prediction_config.enabled:
      prediction_df = self._load_and_merge(self.test_prediction_config.dataset_dir, self.test_prediction_config.dataset_files)
      normalized_prediction_df = self._normalize(prediction_df)
    else:
      normalized_prediction_df = None

    prepared_dataset = PreparedDataset(
      training_df=normalized_training_df,
      validation_df=normalized_validation_df,
      prediction_df=normalized_prediction_df,
      training_ds=self._df_to_np(normalized_training_df),
      validation_ds=self._df_to_np(normalized_validation_df) if normalized_validation_df is not None else None,
      prediction_ds=self._df_to_np(normalized_prediction_df) if normalized_prediction_df is not None else None,
      split_amount=split_amount,
      timestamp_col=self.dataset_config.timestamp_col,
      value_col=self.dataset_config.value_col,
      scaler=self.scaler,
    )

    return prepared_dataset