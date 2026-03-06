from pydantic import BaseModel

class TrainingDatasetConfig(BaseModel):
  windows_size: int
  timestamp_col: str
  value_col: str