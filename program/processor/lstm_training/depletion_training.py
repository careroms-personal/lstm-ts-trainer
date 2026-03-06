import pandas as pd
import numpy as np

from models.config_models import LSTMTsTrainingConfig
from models.trainer_models import TrainingDatasetConfig

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

class DepletionTraining:
  def __init__(self,normalized_data: pd.DataFrame, training_config: LSTMTsTrainingConfig):
    self.dataset_config = TrainingDatasetConfig(
      timestamp_col=training_config.dataset_config.timestamp_col,
      value_col=training_config.dataset_config.value_col,
      windows_size=training_config.training_config.windows_size,
    )

    self.normalized_data = normalized_data

  def _prepare_data(self, windows_size: int):
    values = self.normalized_data[self.dataset_config.value_col].values

    X, y = [], []

    for i in range(values - windows_size):
      X.append(values[i:i + windows_size])
      y.append(values[i + windows_size])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y
  
  def _build_model(self, windows_size: int):
    model = Sequential([
      LSTM(64, return_sequences=True, input_shape=(windows_size, 1)),
      Dropout(0.2),
      LSTM(32, return_sequences=False),
      Dropout(0.2),
      Dense(1),
    ])

    model.compile(
      optimizer="adam",
      loss="mse"
    )

  def train(self):
    pass