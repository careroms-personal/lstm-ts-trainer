import yaml, sys

from pathlib import Path
from pydantic import ValidationError

from models.config_models import LSTMTsTrainingConfig

from .dataset_prepare_executor import DatasetPrepareExecutor
from .training_executor import TrainingExecutor

class Processor:
  def __init__(self, config_path: str):
    self._load_and_validate_config(config_path=config_path)

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
    self.training_executor.execute()