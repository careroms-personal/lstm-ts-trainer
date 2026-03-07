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
      windows_size=training_config.lstm_config.windows_size,
    )

    self.lstm_config = training_config.lstm_config
    self.normalized_data = normalized_data

  def _prepare_data(self, windows_size: int):
    values = self.normalized_data[self.dataset_config.value_col].values

    X, y = [], []

    for i in range(len(values) - windows_size):
      X.append(values[i:i + windows_size])
      y.append(values[i + windows_size])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y
  
  def _build_model(self, windows_size: int):
    units = self.lstm_config.units
    dropout = self.lstm_config.dropout

    model = Sequential([
      LSTM(units[0], return_sequences=True, input_shape=(windows_size, 1)),
      Dropout(dropout),
      LSTM(units[1], return_sequences=False),
      Dropout(dropout),
      Dense(1),
    ])

    model.compile(
      optimizer="adam",
      loss="mse"
    )

    return model

  def train(self):
    windows_size = self.dataset_config.windows_size

    X, y = self._prepare_data(windows_size=windows_size)

    split = int(len(X) * 0.8)

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Total samples: {len(X)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples:   {len(X_val)}")

    model = self._build_model(windows_size=windows_size)

    history = model.fit(
      X_train, y_train,
      epochs=self.lstm_config.epochs,
      batch_size=self.lstm_config.batch_size,
      validation_data=(X_val, y_val),
      shuffle=False,
      verbose=1,
    )

    return model, history