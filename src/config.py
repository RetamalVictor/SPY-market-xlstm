from dataclasses import dataclass
from typing import Optional, Callable, List
import yaml

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    warmup_steps: int
    total_steps: int
    num_features: int
    model_name: str
    output_size:int

@dataclass
class DatasetConfig:
    """
    Description:
        Configuration for the TradingDataset.
    Args:
        file_path (str): Path to the Excel file.
        seq_len (int): Sequence length (window size) to use for inputs.
        mode (str): One of 'train', 'val', or 'test'.
        device (str): Device on which to load the tensors (e.g., 'cpu' or 'cuda').
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
    """
    file_path: str
    seq_len: int
    norm_param_path:str
    mode: str = 'train'
    device:str = "cpu"
    train_ratio: float = 0.7
    val_ratio: float = 0.15

@dataclass
class XLSTMConfig:
    hidden_size: int
    num_heads: int
    # A list specifying the block type for each layer, e.g., ["s", "m", "s"]
    layers: List[str]
    proj_factor_slstm: float
    proj_factor_mlstm: float
    dropout: float

# Mapping from model names (lowercase) to their model-specific configuration dataclasses.
MODEL_CONFIG_MAPPING = {
    "xlstm": XLSTMConfig,
}


def load_config(yaml_file: str):
    """
    Loads training, dataset, and model-specific configurations from a YAML file.
    
    Args:
        yaml_file (str): Path to the YAML configuration file.
        
    Returns:
        Tuple[DatasetConfig, TrainingConfig, object]: The dataset config, training config, and an instance of the model-specific config.
    """
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)
    training_config = TrainingConfig(**cfg["training"])
    dataset_config = DatasetConfig(**cfg["dataset"])
    
    model_name = training_config.model_name.lower()
    model_config_data = cfg.get(model_name, {})
    if model_name in MODEL_CONFIG_MAPPING:
        ModelConfigClass = MODEL_CONFIG_MAPPING[model_name]
        model_config = ModelConfigClass(**model_config_data)
    else:
        raise ValueError(f"Unknown model name '{training_config.model_name}' in config.")
    
    return dataset_config, training_config, model_config

def get_model_class(model_name: str):
    """
    Returns the model class corresponding to the provided model name.
    
    Args:
        model_name (str): Name of the model (e.g., "SimpleNN", "LSTM", or "xLSTM").
        
    Returns:
        The corresponding model class if found.
    """
    model_name = model_name.lower()
    if model_name == "xlstm":
        from src.models.xlstm_model import xLSTMWrapper
        return xLSTMWrapper
    else:
        raise ValueError(f"Unknown model name: {model_name}")
