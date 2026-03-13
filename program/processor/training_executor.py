import pandas as pd
import numpy as np
import tensorflow as tf # type: ignore
import joblib # type: ignore

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from pathlib import Path

from models.dataset_models import PreparedDataset
from models.config_models import LSTMTsTrainingConfig

class TrainingExecutor:
  def __init__(self, prepared_dataset: PreparedDataset, training_config: LSTMTsTrainingConfig):
    self.prepared_dataset = prepared_dataset
    self.training_config = training_config
    self.lstm_config = training_config.lstm_config
    
  def _convert_pd_to_np(self, dataset: pd.DataFrame):
    values = dataset[self.prepared_dataset.value_col].values

    X, y = [], []

    for i in range(len(values) - self.lstm_config.windows_size):
      X.append(values[i:i + self.lstm_config.windows_size])
      y.append(values[i + self.lstm_config.windows_size])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y
  
  def _prepare_training_data(self):
    X_train, y_train = self._convert_pd_to_np(self.prepared_dataset.training_dataset)

    if self.prepared_dataset.validation_dataset is not None:
      X_val, y_val = self._convert_pd_to_np(self.prepared_dataset.validation_dataset)
    else:
      split = int(len(X_train) * self.prepared_dataset.split_amount)

      X_train, X_val = X_train[:split], X_train[split:]
      y_train, y_val = y_train[:split], y_train[split:]

    return X_train, X_val, y_train, y_val
  
  def _build_model(self, windows_size: int):
    units = self.lstm_config.units
    dropout = self.lstm_config.dropout

    tf.keras.backend.set_floatx(self.lstm_config.float_type)

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
  
  def _export_model(self, model):
    export_dir = Path(self.training_config.model_export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    export_path = export_dir / f"{self.training_config.model_name}.keras"
    model.save(export_path)
    print(f"Model exported to: {export_path}")

    scaler_path = export_dir / f"{self.training_config.model_name}_scaler.joblib"
    joblib.dump(self.prepared_dataset.scaler, scaler_path)
    print(f"Scaler exported to: {scaler_path}")

  def execute(self):
    X_train, X_val, y_train, y_val = self._prepare_training_data()
    model = self._build_model(self.lstm_config.windows_size)

    early_stopping = EarlyStopping(
      monitor="val_loss",
      patience=self.lstm_config.patience,
      restore_best_weights=True,
    )

    history = model.fit(
      X_train, y_train,
      epochs=self.lstm_config.epochs,
      batch_size=self.lstm_config.batch_size,
      validation_data=(X_val, y_val),
      shuffle=False,
      verbose=1,
      callbacks=[early_stopping],
    )

    self._export_model(model)

    return model, history