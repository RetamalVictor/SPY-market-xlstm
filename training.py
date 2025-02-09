import os
import glob
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import matplotlib
matplotlib.use("Agg")

from src.dataset import CustomTradingSequenceDataset
from src.config import load_config, get_model_class
from src.models.lightning_wrapper import GenericLightningModule
from src.callbacks.inference_callback import InferencePlotCallback
from pytorch_lightning.callbacks import LearningRateMonitor

def main(config_path: str):
    """
    Description:
        Loads configuration from a YAML file and preprocessed dataset splits (train/val/test)
        from a directory specified in the config, instantiates a model, sets up DataLoaders and callbacks 
        (including an inference callback that runs on the test set), and starts training.
    
    Args:
        config_path (str): Path to the YAML configuration file (e.g., train_config.yaml).
    
    Returns:
        None
    """
    # Load training and dataset configuration from YAML.
    dataset_config, training_config, model_config = load_config(config_path)
    
    # Use the data directory from the training configuration.
    data_dir = training_config.data_dir
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")
    
    # Locate the pre-split files in the data directory.
    train_files = glob.glob(os.path.join(data_dir, "*_train.parquet"))
    val_files = glob.glob(os.path.join(data_dir, "*_val.parquet"))
    test_files = glob.glob(os.path.join(data_dir, "*_test.parquet"))

    if not train_files or not val_files or not test_files:
        raise FileNotFoundError("Could not find one or more of the required files: *_train.parquet, *_val.parquet, *_test.parquet in the directory: " + data_dir)
    
    train_file = train_files[0]
    val_file = val_files[0]
    test_file = test_files[0]
    
    print("Train file:", train_file)
    print("Validation file:", val_file)
    print("Test file:", test_file)
    
    # Load pre-split datasets.
    train_dataset = CustomTradingSequenceDataset(train_file, dataset_config)
    val_dataset = CustomTradingSequenceDataset(val_file, dataset_config)
    test_dataset = CustomTradingSequenceDataset(test_file, dataset_config)
    
    print("Train dataset valid samples:", len(train_dataset))
    print("Validation dataset valid samples:", len(val_dataset))
    print("Test dataset valid samples:", len(test_dataset))
    
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Calculate input size: for this example, we flatten the input sequence.
    # The dataset has 7 features and the sequence length (in bars) is computed as:
    # sequence_length_bars = dataset_config.sequence_length_minutes // dataset_config.time_interval_minutes
    # input_size = sequence_length_bars * training_config.num_features
    input_size = training_config.num_features
    output_size = training_config.output_size
    
    # Retrieve the model class using model mapping.
    model_cls = get_model_class(training_config.model_name)
    # Instantiate the underlying torch model using the model-specific configuration via the class method.
    torch_model = model_cls.from_config(input_size, output_size, model_config)
    scripted_model = torch.jit.script(torch_model)

    # Wrap the torch model with the generic Lightning wrapper, enabling cosine warmup if configured
    model = GenericLightningModule(
        model=scripted_model,
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
    inference_callback = InferencePlotCallback(test_loader, num_batches=1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Set up a ModelCheckpoint callback to save the best model based on validation loss.
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=2,
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
        description="Training script using pre-split train/val/test files from a directory."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file (e.g., train_config.yaml)")
    args = parser.parse_args()
    main(args.config)