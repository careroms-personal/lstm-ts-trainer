from pydantic import BaseModel, model_validator
from enum import StrEnum
from typing import Optional, List

class PredictionType(StrEnum):
  Depletion = "depletion"
  Incremental = "incremental"

class DataSetConfig(BaseModel):
  dataset_dir: str
  dataset_files: List[str] = ["*"]
  timestamp_col: str
  value_col: str

class PredictionBoundary(BaseModel):
  floor_value: float = 0.0
  ceiling_value: float = 0.0

class TrainingConfig(BaseModel):
  windows_size: int = 24
  prediction_boundary: PredictionBoundary

  @model_validator(mode='after')
  def validate_boundary(self):
    if self.prediction_boundary.ceiling_value <= self.prediction_boundary.floor_value:
        raise ValueError("ceiling_value must be greater than floor_value")
    if self.prediction_boundary.ceiling_value == 0.0:
        raise ValueError("ceiling_value cannot be zero")
    return self

class TransferTraining(BaseModel):
  base_dir: str
  model_list: List[str] = ["*"]

class LSTMTsTrainingConfig(BaseModel):
  prediction_type: PredictionType
  model_name: str
  model_export_dir: str
  dataset_config: DataSetConfig
  training_config: TrainingConfig
  transfer_training: Optional[TransferTraining] = None