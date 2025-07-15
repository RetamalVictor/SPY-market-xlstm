import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataset_spy import TradingDataset
from src.config import load_config, get_model_class
from src.models.lightning_wrapper_multitask import GenericLightningModule
from src.callbacks.inference_callback import InferencePlotCallback

def main(config_path: str):
    """
    Description:
        Loads configuration from a YAML file, sets up dataset splits, instantiates a model, and starts training.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        None
    """
    dataset_config, training_config, model_config = load_config(config_path)
    
    # Set multi_task flag for dataset and adjust output_size if multi-task mode is enabled.
    if getattr(dataset_config, "multi_task", False):
        training_config.output_size = 2

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

    sample_input, _ = train_dataset[0]
    input_size = sample_input.shape[-1]
    output_size = training_config.output_size

    model_cls = get_model_class(training_config.model_name)
    torch_model = model_cls.from_config(input_size, output_size, model_config)
    scripted_model = torch.jit.script(torch_model)

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
    
    hyperparams = {**vars(training_config), **vars(dataset_config)}
    model.save_hyperparameters(hyperparams)
    
    logger = TensorBoardLogger("tb_logs", name="spy_xlstm_multitask")
    inference_callback = InferencePlotCallback(test_loader, num_batches=1)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="best-{epoch:02d}-{val_f1:.2f}"
    )

    trainer = pl.Trainer(
        max_epochs=training_config.epochs,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[inference_callback, checkpoint_callback, lr_monitor],
        log_every_n_steps=1
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script using pre-split train/val/test files from a directory."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file (e.g., train_config.yaml)")
    args = parser.parse_args()
    main(args.config)
