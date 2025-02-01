import math
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataset import CustomTradingSequenceDataset
from src.config import DatasetConfig
from src.models.baseline import BaselineLightingModel

# Hyperparameters
BATCH_SIZE = 64
HIDDEN_SIZE = 32
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
EPOCHS = 5  # Adjust as needed
DATA_FILE = "data\\oro_transformed.parquet"  # Replace with your actual data file path

def main():
    """
    Description: Main function to set up data, model, and training.
    Args:
        None
    Raises:
        Exception: If dataset loading or splitting fails.
    Return:
        None
    """
    # Dataset configuration: adjust values as needed
    config = DatasetConfig(
        sequence_length_minutes=60, 
        prediction_horizon='next_half_hour', 
        time_interval_minutes=5
    )
    full_dataset = CustomTradingSequenceDataset(DATA_FILE, config)
    
    train_size = math.floor(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # The dataset features have 7 columns, so input_size = 7.
    input_size = 7
    
    model = BaselineLightingModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        learning_rate=LEARNING_RATE
    )
    
    logger = TensorBoardLogger("tb_logs", name="lighting_model")
    
    # Use accelerator and devices based on GPU availability
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            logger=logger,
            accelerator="gpu",
            devices=1
        )
    else:
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            logger=logger
        )
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
