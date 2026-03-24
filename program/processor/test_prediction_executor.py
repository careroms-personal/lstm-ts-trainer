import numpy as np # type: ignore
import pandas as pd
import tensorflow as tf # type: ignore

from pathlib import Path

from models.dataset_models import PreparedDataset
from models.config_models import LSTMTsTrainingConfig

class TestPredictionExecutor:
  def __init__(self, prepare_dataset: PreparedDataset, model: tf.keras.Model, training_config: LSTMTsTrainingConfig):
    self.prepare_dataset = prepare_dataset
    self.model = model
    self.training_config = training_config

  def _test_prediction(self):
    X_pred = self.prepare_dataset.prediction_ds.X

    y_pred_scaled = self.model.predict(X_pred)
    y_true_scaled = self.prepare_dataset.prediction_ds.y.reshape(-1, 1)

    scaler = self.prepare_dataset.scaler
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_true_scaled)

    return y_pred.flatten(), y_true.flatten()

  def _compute_metrics(self, y_pred: np.ndarray, y_true: np.ndarray):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    return mae, rmse

  def _export_results(self, y_pred: np.ndarray, y_true: np.ndarray, mae: float, rmse: float):
    output_config = self.training_config.dataset_config.test_prediction_config.output_config

    if output_config.print_output:
      print(f"Test Prediction Results:")
      print(f"  MAE:  {mae:.4f}")
      print(f"  RMSE: {rmse:.4f}")

    if output_config.export_output.enabled:
      export_dir = Path(output_config.export_output.export_dir)
      export_dir.mkdir(parents=True, exist_ok=True)

      result_df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
      csv_path = export_dir / f"{self.training_config.model_name}_test_prediction.csv"
      result_df.to_csv(csv_path, index=False)
      print(f"  Results exported to: {csv_path}")

  def execute(self):
    if self.training_config.dataset_config.test_prediction_config.enabled == False:
      return

    y_pred, y_true = self._test_prediction()
    mae, rmse = self._compute_metrics(y_pred, y_true)
    self._export_results(y_pred, y_true, mae, rmse)
