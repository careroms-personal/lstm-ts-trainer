import yaml, sys
import tensorflow as tf # type: ignore

from pathlib import Path
from pydantic import ValidationError

from models.config_models import LSTMTsTrainingConfig

from .dataset_prepare_executor import DatasetPrepareExecutor
from .training_executor import TrainingExecutor
from .test_prediction_executor import TestPredictionExecutor

class Processor:
  def __init__(self, config_path: str):
    self._configure_hardware()
    self._load_and_validate_config(config_path=config_path)

  def _configure_hardware(self):
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
      print("No GPU detected, using CPU")
      return

    try:
      tf.config.experimental.set_memory_growth(gpus[0], True)
      print(f"Using GPU: {gpus[0].name}")
        
        # Add these for confirmation
      print(f"CUDA available: {tf.test.is_built_with_cuda()}")
      print(f"GPU available: {tf.test.is_gpu_available()}")  
      print(f"Devices: {tf.config.list_logical_devices()}")
    except RuntimeError as e:
      print(f"GPU configuration error: {e}")

  def _load_and_validate_config(self, config_path: str):
    if not Path(config_path).exists():
      print(f"❌ Config file not found: {config_path}")
      sys.exit(1)

    try:
      with open(config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
        self.lstm_ts_training_config = LSTMTsTrainingConfig(**yaml_data)

    except ValidationError as e:
      print(f"❌ Invalid config file:")

      for error in e.errors():
        print(f"   - {error['loc']}: {error['msg']}")

      sys.exit(1)

  def execute(self):
    self.dataset_prepare_executor = DatasetPrepareExecutor(self.lstm_ts_training_config)
    self.training_dataset = self.dataset_prepare_executor.execute()

    self.training_executor = TrainingExecutor(self.training_dataset, self.lstm_ts_training_config)
    model, _ = self.training_executor.execute()

    self.test_prediction_executor = TestPredictionExecutor(self.training_dataset, model, self.lstm_ts_training_config)
    self.test_prediction_executor.execute()
