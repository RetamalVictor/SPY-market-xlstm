from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class DatasetConfig:
    """
    Configuration parameters for the CustomTradingSequenceDataset.
    """
    sequence_length_minutes: int     # e.g., 60 minutes of data for the input sequence
    prediction_horizon: str         # 'next_half_hour', 'next_hour', 'next_day'
    time_interval_minutes: int      # e.g., 5 if each bar is 5 minutes
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
