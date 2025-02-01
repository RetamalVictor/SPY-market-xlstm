from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import yaml

@dataclass
class DatasetConfig:
    """
    Configuration parameters for the CustomTradingSequenceDataset.
    """
    sequence_length_minutes: int     # e.g., 60 minutes of data for the input sequence
    prediction_horizon: str            # 'next_half_hour', 'next_hour', 'next_day'
    time_interval_minutes: int         # e.g., 5 if each bar is 5 minutes
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None

@dataclass
class TrainingConfig:
    """
    Configuration parameters for training.
    """
    batch_size: int
    hidden_size: int
    num_layers: int
    learning_rate: float
    epochs: int
    data_file: str
    warmup_steps: int = 0
    total_steps: int = 0

def load_config(yaml_file: str) -> Tuple[DatasetConfig, TrainingConfig]:
    """
    Description:
        Loads training and dataset configuration from a YAML file.
    Args:
        yaml_file (str): Path to the YAML configuration file.
    Raises:
        Exception: If the file cannot be loaded or parsed.
    Return:
        Tuple[DatasetConfig, TrainingConfig]: The dataset and training configurations.
    """
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)
    dataset_config = DatasetConfig(**cfg["dataset"])
    training_config = TrainingConfig(**cfg["training"])
    return dataset_config, training_config
