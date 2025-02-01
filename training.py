import argparse
import math
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import matplotlib
matplotlib.use("Agg")

from src.dataset import CustomTradingSequenceDataset
from src.config import load_config
from src.models.baseline import SimpleNN
from src.models.lightning_wrapper import GenericLightningModule
from src.callbacks.inference_callback import InferencePlotCallback
from pytorch_lightning.callbacks import LearningRateMonitor

def main(config_path: str):
    """
    Description:
        Main function to load configuration, set up dataset, model, training, and callbacks.
    Args:
        config_path (str): Path to the YAML configuration file.
    Raises:
        Exception: If dataset loading or splitting fails.
    Return:
        None
    """
    # Load training and dataset configuration from YAML file
    dataset_config, training_config = load_config(config_path)

    # Initialize dataset using the training config's data_file and dataset config
    full_dataset = CustomTradingSequenceDataset(training_config.data_file, dataset_config)

    # Split dataset into training and validation sets (90% train, 10% val)
    train_size = math.floor(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Calculate input size: for this example, we flatten the input sequence.
    # The dataset has 7 features and the sequence length (in bars) is computed as:
    sequence_length_bars = dataset_config.sequence_length_minutes // dataset_config.time_interval_minutes
    input_size = sequence_length_bars * 7

    # Instantiate a simple neural network model that extends BaseModel (see src/models/simple_nn.py)
    torch_model = SimpleNN(
        input_size=input_size,
        hidden_size=training_config.hidden_size,
        output_size=1
    )

    # Wrap the torch model with the generic Lightning wrapper, enabling cosine warmup if configured
    model = GenericLightningModule(
        model=torch_model,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={'lr': training_config.learning_rate},
        loss_fn=torch.nn.MSELoss(),
        warmup_steps=training_config.warmup_steps,
        total_steps=training_config.total_steps
    )
    
    # Merge all hyperparameters (both training and dataset) into a single dictionary
    hyperparams = {**vars(training_config), **vars(dataset_config)}
    model.save_hyperparameters(hyperparams)
    # Set up TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="lighting_model")

    # Create the inference callback that logs predictions vs. real targets after validation
    inference_callback = InferencePlotCallback(val_loader, num_batches=1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Set up a ModelCheckpoint callback to save the best model based on validation loss.
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.2f}"
    )

    # Initialize the PyTorch Lightning trainer with GPU support if available
    trainer = pl.Trainer(
        max_epochs=training_config.epochs,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[inference_callback, checkpoint_callback, lr_monitor]
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script for the Lightning model with inference and checkpointing"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g., train_config.yaml)"
    )
    args = parser.parse_args()
    main(args.config)