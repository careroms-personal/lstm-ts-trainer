from pydantic import BaseModel, model_validator
from typing import Optional, List, Literal

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

class ExportOutputConfig(BaseModel):
  enabled: bool = False
  export_dir: str = ""

class OutputConfig(BaseModel):
  print_output: bool = True
  export_output: ExportOutputConfig = ExportOutputConfig()

class TestPredictionConfig(BaseModel):
  enabled: bool = False
  output_config: OutputConfig = OutputConfig()
  dataset_dir: str = ""
  dataset_files: List[str] = ["*"]

class DataSetConfig(BaseModel):
  training_data: TrainingDataConfig
  validation_config: ValidationConfig = ValidationConfig()
  test_prediction_config: TestPredictionConfig = TestPredictionConfig()
  timestamp_col: str
  value_col: str

class LSTMConfig(BaseModel):
  windows_size: int = 24
  units: List[int] = [64, 32]
  dropout: float = 0.2
  epochs: int = 50
  batch_size: int = 32
  patience: int = 10
  float_type: Literal["float32", "float64"] = "float32"

class TransferTraining(BaseModel):
  base_dir: str
  model_list: List[str] = ["*"]

class LSTMTsTrainingConfig(BaseModel):
  model_name: str
  model_export_dir: str
  dataset_config: DataSetConfig
  lstm_config: LSTMConfig
  transfer_training: Optional[TransferTraining] = None
