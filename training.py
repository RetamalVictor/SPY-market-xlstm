import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataset_spy import TradingDataset  # now using the new DatasetConfig version
from src.config import load_config, get_model_class
from src.models.lightning_wrapper_classification import GenericLightningModule
from src.callbacks.inference_callback import InferencePlotCallback

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
    # Load training, dataset, and model configuration from YAML.
    dataset_config, training_config, model_config = load_config(config_path)
    
    # Create dataset splits using the new dataset configuration.
    dataset_config.mode = "train"
    train_dataset = TradingDataset(dataset_config)
    dataset_config.mode = "val"
    val_dataset = TradingDataset(dataset_config)
    dataset_config.mode = "test"
    test_dataset = TradingDataset(dataset_config)
    
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
        num_workers=4,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Determine the actual input size from the training dataset.
    # Each sample returned by the dataset is of shape (sequence_length_bars, actual_feature_dim)
    sample_input, _ = train_dataset[0]
    # sample_input shape: [seq_len, feature_dim]
    input_size = sample_input.shape[-1]
    output_size = training_config.output_size

    # Retrieve the model class using model mapping.
    model_cls = get_model_class(training_config.model_name)
    # Instantiate the underlying torch model using the model-specific configuration via the class method.
    torch_model = model_cls.from_config(input_size, output_size, model_config)
    scripted_model = torch.jit.script(torch_model)

    # Wrap the torch model with the generic Lightning wrapper.
    model = GenericLightningModule(
        model=scripted_model,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={
            "lr": float(training_config.learning_rate),
            "weight_decay": float(training_config.weight_decay)
        },        
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        warmup_steps=training_config.warmup_steps,
        total_steps=training_config.total_steps
    )
    
    # Merge all hyperparameters (both training and dataset) into a single dictionary.
    hyperparams = {**vars(training_config), **vars(dataset_config)}
    model.save_hyperparameters(hyperparams)
    
    # Set up TensorBoard logger.
    logger = TensorBoardLogger("tb_logs", name="spy_xlstm_clas")

    # Create the inference callback that logs predictions vs. real targets after validation.
    inference_callback = InferencePlotCallback(test_loader, num_batches=1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Set up a ModelCheckpoint callback to save the best model based on validation loss.
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="best-{epoch:02d}-{val_f1:.2f}"
    )

    # Initialize the PyTorch Lightning trainer with GPU support if available.
    trainer = pl.Trainer(
        max_epochs=training_config.epochs,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[inference_callback, checkpoint_callback, lr_monitor],
        log_every_n_steps=1
    )

    # Start training.
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script using pre-split train/val/test files from a directory."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file (e.g., train_config.yaml)")
    args = parser.parse_args()
    main(args.config)
