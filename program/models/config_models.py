from pydantic import BaseModel, model_validator
from typing import Optional, List

class ExternalDataConfig(BaseModel):
  enabled: bool = False
  dataset_dir: str = ""
  dataset_files: List[str] = ["*"]

class InternalDataConfig(BaseModel):
  split_amount: float = 0.2

class ValidationConfig(BaseModel):
  external_data: ExternalDataConfig = ExternalDataConfig()
  internal_data: Optional[InternalDataConfig] = InternalDataConfig()

  @model_validator(mode='after')
  def disable_internal_if_external_enabled(self):
    if self.external_data.enabled:
      self.internal_data = None
    return self

class TrainingDataConfig(BaseModel):
  dataset_dir: str
  dataset_files: List[str] = ["*"]

class DataSetConfig(BaseModel):
  training_data: TrainingDataConfig
  validation_config: ValidationConfig = ValidationConfig()
  timestamp_col: str
  value_col: str

class PredictionBoundary(BaseModel):
  floor_value: float = 0.0
  ceiling_value: float = 0.0

class LSTMConfig(BaseModel):
  windows_size: int = 24
  units: List[int] = [64, 32]
  dropout: float = 0.2
  epochs: int = 50
  batch_size: int = 32
  patience: int = 10

class TransferTraining(BaseModel):
  base_dir: str
  model_list: List[str] = ["*"]

class LSTMTsTrainingConfig(BaseModel):
  model_name: str
  model_export_dir: str
  dataset_config: DataSetConfig
  prediction_boundary: PredictionBoundary
  lstm_config: LSTMConfig
  transfer_training: Optional[TransferTraining] = None

  @model_validator(mode='after')
  def validate_boundary(self):
    if self.prediction_boundary.ceiling_value <= self.prediction_boundary.floor_value:
        raise ValueError("ceiling_value must be greater than floor_value")
    if self.prediction_boundary.ceiling_value == 0.0:
        raise ValueError("ceiling_value cannot be zero")
    return self