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
from src.dataset import CustomTradingSequenceDataset
from src.config import get_model_class, load_config
from src.models.baseline import SimpleNN  # Underlying torch model for inference.
from src.models.lightning_wrapper import GenericLightningModule

def run_inference(checkpoint: str, config_path: str, data_dir: str, evaluate: bool, batch_size: int = 64, num_samples: int = 100):
    """
    Description:
        Loads configuration from YAML, finds the test file in the given data directory,
        instantiates a test dataset and DataLoader from that test file, loads a trained model checkpoint using 
        the class method (providing the underlying torch model), and runs inference.
        If evaluate is set, it plots predictions vs. real targets.
    
    Args:
        checkpoint (str): Path to the trained model checkpoint.
        config_path (str): Path to the YAML configuration file (e.g., train_config.yaml).
        data_dir (str): Directory containing the preprocessed test file.
        evaluate (bool): If True, plots predictions vs. real targets; otherwise, prints them.
        batch_size (int, optional): Batch size for inference.
        num_samples (int, optional): Number of samples to run inference on.
    
    Returns:
        None
    """
    # Load configuration.
    dataset_config, training_config, model_config = load_config(config_path)
    
    # Use the data directory from the config.
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")
    
    # Locate the test file in the data directory.
    test_files = glob.glob(os.path.join(data_dir, "*_test.parquet"))
    if not test_files:
        test_files = glob.glob(os.path.join(data_dir, "*_test.csv"))
    if not test_files:
        raise FileNotFoundError("No test file found in the directory (expecting *_test.parquet or *_test.csv).")
    test_file = test_files[0]
    print("Using test file:", test_file)
    
    # Load the test dataset.
    test_dataset = CustomTradingSequenceDataset(test_file, dataset_config)
    print("Test dataset valid samples:", len(test_dataset))
    
    # Optionally restrict to a subset of samples.
    if num_samples < len(test_dataset):
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, list(range(num_samples)))
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Compute sequence length in bars.
    sequence_length_bars = dataset_config.sequence_length_minutes // dataset_config.time_interval_minutes
    # Compute model input size: number of bars * num_features (from YAML, e.g., 8).
    input_size = sequence_length_bars * training_config.num_features
    print("Computed input size for model:", input_size)
    
    # Retrieve the model class using model mapping.
    model_cls = get_model_class(training_config.model_name)
    # Instantiate the underlying torch model using the model-specific configuration via the class method.
    torch_model = model_cls.from_config(input_size, model_config)   
    
    # Load the Lightning model checkpoint, providing the underlying model.
    model = GenericLightningModule.load_from_checkpoint(
        checkpoint_path=checkpoint,
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
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            preds = model(inputs)
            predictions.extend(preds.cpu().tolist())
            targets.extend(target.cpu().tolist())
    
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
        description="Inference script for a trained model using preprocessed test files from a directory specified in the config."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file (e.g., train_config.yaml)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the preprocessed test file (e.g., *_test.parquet)")
    parser.add_argument("--evaluate", action="store_true",
                        help="If set, plots predictions vs. real targets; otherwise, prints predictions.")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to run inference on (default: 100)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference (default: 64)")
    args = parser.parse_args()
    run_inference(args.checkpoint, args.config, args.data_dir, args.evaluate, args.batch_size, args.num_samples)

if __name__ == "__main__":
    main()
