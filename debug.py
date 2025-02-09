#!/usr/bin/env python
import argparse
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import CustomTradingSequenceDataset
from src.config import load_config, get_model_class

def debug_loss(config_path: str, data_file: str):
    """
    Description:
        Loads configuration and dataset from a given file, instantiates a simple model,
        performs one forward pass with one batch, computes the loss, and prints it.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        data_file (str): Path to the preprocessed dataset file (CSV or Parquet).
    
    Raises:
        Exception: If any step fails.
    
    Returns:
        None
    """
    # Load configuration.
    dataset_config, training_config, model_config = load_config(config_path)
    
    # Instantiate the dataset.
    dataset = CustomTradingSequenceDataset(data_file, dataset_config)
    print(f"File: {os.path.basename(data_file)} - Number of valid samples in dataset: {len(dataset)}")
    
    # Create a DataLoader for one batch (using num_workers=0 for simplicity).
    dataloader = DataLoader(dataset, batch_size=training_config.batch_size, shuffle=False, num_workers=0)
    
    # Get one batch from the dataloader.
    batch = next(iter(dataloader))
    inputs, targets = batch
    print("  Batch input shape:", inputs.shape)
    print("  Batch target shape:", targets.shape)
    
    # Check for NaNs/Infs in inputs and targets.
    if torch.isnan(inputs).any():
        print("  Found NaN in inputs!")
    if torch.isinf(inputs).any():
        print("  Found Inf in inputs!")
    if torch.isnan(targets).any():
        print("  Found NaN in targets!")
    if torch.isinf(targets).any():
        print("  Found Inf in targets!")
    
    # Determine the input size for the model.
    sequence_length, num_features = inputs.shape[1], inputs.shape[2]
    input_size = num_features
    print("  Computed input size for model:", input_size)
    
    # Instantiate the model.
    # Retrieve the model class using model mapping.
    model_cls = get_model_class(training_config.model_name)
    # Instantiate the underlying torch model using the model-specific configuration via the class method.
    torch_model = model_cls.from_config(input_size, model_config)    
    # Flatten inputs to match the model's expected shape (batch_size, input_size).
    # inputs_flat = inputs.view(inputs.size(0), -1)
    
    # Compute model outputs.
    outputs = torch_model(inputs)
    
    # Print output statistics.
    print("  Output stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        outputs.min().item(), outputs.max().item(), outputs.mean().item()
    ))
    if torch.isnan(outputs).any():
        print("  NaN detected in model outputs!")
    
    outputs = outputs.squeeze(-1)
    
    # Compute the loss.
    loss_fn = nn.MSELoss()
    loss = loss_fn(outputs, targets)
    
    print("  Debug Loss:", loss.item())
    if torch.isnan(loss):
        print("  Loss is NaN!")

def main(config_path: str, data_dir: str):
    """
    Description:
        Iterates over all processed dataset files (CSV or Parquet) in the provided directory
        and runs the debug loss computation on each.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        data_dir (str): Directory containing the processed dataset files.
    
    Returns:
        None
    """
    # Get list of files in the data_dir matching CSV or Parquet.
    file_patterns = [os.path.join(data_dir, "*.parquet"), os.path.join(data_dir, "*.csv")]
    data_files = []
    for pattern in file_patterns:
        data_files.extend(glob.glob(pattern))
    
    if not data_files:
        raise FileNotFoundError(f"No processed dataset files found in directory: {data_dir}")
    
    print(f"Found {len(data_files)} files in {data_dir}.")
    
    # Iterate over each file and run debug_loss.
    for data_file in data_files:
        print("\n-----------------------------------------")
        debug_loss(config_path, data_file)
        print("-----------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug loss computation on one batch for all processed files in a directory."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file (e.g., train_config.yaml)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the preprocessed data files (CSV or Parquet)")
    args = parser.parse_args()
    main(args.config, args.data_dir)
