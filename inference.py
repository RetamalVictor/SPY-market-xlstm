#!/usr/bin/env python
import argparse
import glob
import os
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import project modules.
from src.dataset_spy import TradingDataset  # New dataset
from src.config import get_model_class, load_config, DatasetConfig
from src.models.baseline import SimpleNN  # Underlying torch model for inference.
from src.models.lightning_wrapper import GenericLightningModule

def run_inference(checkpoint: str, config_path: str, evaluate: bool, batch_size: int = 10, num_samples: int = 100):
    """
    Description:
        Loads configuration from YAML, locates the test file in the given data directory,
        instantiates a test dataset (using the new TradingDataset) and DataLoader,
        and loads a trained model checkpoint if available. If no checkpoint exists, a new model is created.
        Inference is then run on the test dataset. If evaluate is set, the predictions are plotted against the real targets.
    
    Args:
        checkpoint (str): Path to the trained model checkpoint.
        config_path (str): Path to the YAML configuration file (e.g., train_config.yaml).
        data_dir (str): Directory containing the test file.
        evaluate (bool): If True, plots predictions vs. real targets; otherwise, prints them.
        batch_size (int, optional): Batch size for inference.
        num_samples (int, optional): Number of samples to run inference on.
    
    Returns:
        None
    """
    # Load configuration.
    dataset_config, training_config, model_config = load_config(config_path)

    dataset_config.mode = "test"
    
    # Load the test dataset using the new TradingDataset.
    test_dataset = TradingDataset(dataset_config)
    print("Test dataset valid samples:", len(test_dataset))
    
    # # Optionally restrict to a subset of samples.
    if num_samples < len(test_dataset):
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, list(range(num_samples)))
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Determine input feature dimension dynamically.
    sample, _ = test_dataset[0]
    input_size = sample.shape[-1]
    print("Computed input feature dimension for model:", input_size)
    
    # Retrieve the model class using model mapping.
    model_cls = get_model_class(training_config.model_name)
    # Instantiate the underlying torch model using the model-specific configuration.
    # For models like xLSTMWrapper, from_config requires input_size, output_size, and config.
    torch_model = model_cls.from_config(input_size, training_config.output_size, model_config)
    
    # If a valid checkpoint exists, load the Lightning model checkpoint.
    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading model from checkpoint: {checkpoint}")
        model = GenericLightningModule.load_from_checkpoint(
            checkpoint_path=checkpoint,
            model=torch_model,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={'lr': training_config.learning_rate},
            loss_fn=torch.nn.MSELoss(),
            warmup_steps=training_config.warmup_steps,
            total_steps=training_config.total_steps
        )
    else:
        print("No valid checkpoint provided; creating a new model instance.")
        model = GenericLightningModule(
            model=torch_model,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={'lr': training_config.learning_rate},
            loss_fn=torch.nn.MSELoss(),
            warmup_steps=training_config.warmup_steps,
            total_steps=training_config.total_steps
        )
    model.eval()
    device = next(model.parameters()).device
    model.to(device)
    
    # for batch_idx, (inputs, targets) in enumerate(test_loader):
    #     print(f"Batch {batch_idx}:")
    #     print("  Inputs shape:", inputs.shape)
    #     print("  Targets shape:", targets.shape)
    #     print(f"inputs are {inputs}")
    #     print(f"Target are  {targets}")
    #     # Process only a few batches for demonstration.
    #     if batch_idx >= 3:
    #         break
    
    predictions = []
    targets = []  # Python list for collecting targets

    with torch.no_grad():
        for inputs, target_batch in test_loader:
            inputs = inputs.to(device)
            target_batch = target_batch.to(device)
            preds, _ = model(inputs)
            # Extract final time step predictions if necessary:
            preds = preds[:, -1, :].squeeze(-1)
            predictions.extend(preds.cpu().tolist())
            targets.extend(target_batch.cpu().tolist())

        
    if evaluate:
        plt.figure(figsize=(10, 5))
        plt.plot(targets, label="Real Target")
        plt.plot(predictions, label="Predicted")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.title("Real vs. Predicted Targets")
        plt.legend()
        plt.savefig("inference_results.png", dpi=150)
        plt.close()
        print("Inference plot saved as 'inference_results.png'.")
    else:
        for idx, pred in enumerate(predictions):
            print(f"Sample {idx}: Predicted: {pred}, Real: {targets[idx]}")

def main():
    parser = argparse.ArgumentParser(
        description="Inference script for a trained model using a preprocessed test file from a directory specified in the config."
    )
    parser.add_argument("--checkpoint", type=str, required=False, default=r"tb_logs\spy_xlstm\version_2\checkpoints\best-epoch=03-val_loss=0.01.ckpt",
                        help="Path to the trained model checkpoint. If not provided, a new model will be created.")
    parser.add_argument("--config", type=str, required=False, default="configs/train_config.yaml",
                        help="Path to YAML configuration file (e.g., train_config.yaml)")
    parser.add_argument("--evaluate", action="store_true",
                        help="If set, plots predictions vs. real targets; otherwise, prints predictions.")
    parser.add_argument("--num_samples", type=int, default=900,
                        help="Number of samples to run inference on (default: 100)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference (default: 64)")
    args = parser.parse_args()
    run_inference(args.checkpoint, args.config, args.evaluate, args.batch_size, args.num_samples)

if __name__ == "__main__":
    main()
