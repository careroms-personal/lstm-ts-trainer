import pandas as pd
import numpy as np

from pydantic import BaseModel, model_validator, ConfigDict
from sklearn.preprocessing import MinMaxScaler # type: ignore
from typing import Optional

class NPDataset(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)
  X: np.ndarray
  y: np.ndarray

class PreparedDataset(BaseModel):
  training_df: pd.DataFrame
  validation_df: Optional[pd.DataFrame] = None
  prediction_df: Optional[pd.DataFrame] = None
  training_ds: Optional[NPDataset] = None
  validation_ds: Optional[NPDataset] = None
  prediction_ds: Optional[NPDataset] = None
  split_amount: Optional[float] = None
  timestamp_col: str
  value_col: str
  scaler: MinMaxScaler

  model_config = ConfigDict(arbitrary_types_allowed=True)

  @model_validator(mode='after')
  def validate_validation_source(self):
    if self.validation_df is not None and self.split_amount is not None:
      raise ValueError("Cannot have both validation_df and split_amount!")
    if self.validation_df is None and self.split_amount is None:
      raise ValueError("Must have either validation_df or split_amount!")
    return self