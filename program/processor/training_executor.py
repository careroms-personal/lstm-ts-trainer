import pandas as pd

from models.config_models import LSTMTsTrainingConfig, PredictionType
from .lstm_training.depletion_training import DepletionTraining

class TrainingExecutor:
  def __init__(self, normalized_data: pd.DataFrame ,training_config: LSTMTsTrainingConfig):
    self.training_config = training_config
    self.normalized_data = normalized_data

  def _training_model(self):
    match self.training_config.prediction_type:
      case PredictionType.Depletion:
        training_class = DepletionTraining(
          normalized_data=self.normalized_data,
          training_config=self.training_config
        )
      case PredictionType.Incremental:
        raise NotImplementedError("Incremental prediction type is not yet implemented")

    training_class.train()

  def execute(self):
    return self._training_model()