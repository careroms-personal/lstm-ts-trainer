import pandas as pd
import numpy as np

from pydantic import BaseModel, model_validator, ConfigDict
from typing import Optional

class PreparedDataset(BaseModel):
  training_dataset: pd.DataFrame
  validation_dataset: Optional[pd.DataFrame] = None
  split_amount: Optional[float] = None
  floor_value: float
  ceiling_value: float
  timestamp_col: str
  value_col: str

  model_config = ConfigDict(arbitrary_types_allowed=True)  # allow pd.DataFrame!

  @model_validator(mode='after')
  def validate_validation_source(self):
    if self.validation_dataset is not None and self.split_amount is not None:
      raise ValueError("Cannot have both validation_dataset and split_amount!")
    if self.validation_dataset is None and self.split_amount is None:
      raise ValueError("Must have either validation_dataset or split_amount!")
    if self.ceiling_value <= self.floor_value:
      raise ValueError("ceiling_value must be greater than floor_value!")
    return self