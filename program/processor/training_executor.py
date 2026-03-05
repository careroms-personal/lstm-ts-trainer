from models.config_models import LSTMTsTrainingConfig, PredictionType

class TrainingExecutor:
  def __init__(self, training_config: LSTMTsTrainingConfig):
    self.training_config = training_config

  def _training_model(self):
    match self.training_config.prediction_type:
      case PredictionType.Depletion:
        pass
      case PredictionType.Incremental:
        pass

  def execute(self):
    return self._training_model()