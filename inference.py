import argparse
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from src.dataset import CustomTradingSequenceDataset
from src.config import DatasetConfig
from src.models.baseline import BaselineLightingModel

def run_inference(checkpoint: str, data_file: str, evaluate: bool, batch_size: int = 64, num_samples: int = 100):
    """
    Description:
        Runs inference using a trained model. If the evaluate flag is set,
        the function plots the real target values against the predicted values.
    Args:
        checkpoint (str): Path to the trained model checkpoint.
        data_file (str): Path to the data file for inference.
        evaluate (bool): If True, plot predictions vs. real targets.
        batch_size (int, optional): Batch size for inference. Default is 64.
        num_samples (int, optional): Number of samples to run inference on. Default is 100.
    Raises:
        Exception: If dataset loading or model checkpoint loading fails.
    Return:
        None
    """
    # Create the dataset using the configuration settings
    config = DatasetConfig(
        sequence_length_minutes=60, 
        prediction_horizon='next_half_hour', 
        time_interval_minutes=5
    )
    dataset = CustomTradingSequenceDataset(data_file, config)
    
    # Optionally, restrict the inference to a subset of samples
    if num_samples < len(dataset):
        dataset = Subset(dataset, list(range(num_samples)))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Load the model from checkpoint using the same parameters as during training.
    model = BaselineLightingModel.load_from_checkpoint(
        checkpoint_path=checkpoint,
        input_size=7,       # The dataset features have 7 columns
        hidden_size=32,
        num_layers=1,
        learning_rate=1e-3
    )
    model.eval()

    # Determine the device from the model parameters
    device = next(model.parameters()).device
    model.to(device)

    predictions = []
    targets = []
    
    # Run inference on the data
    with torch.no_grad():
        for inputs, target in dataloader:
            # Move inputs and targets to the same device as the model
            inputs = inputs.to(device)
            target = target.to(device)
            preds = model(inputs)
            predictions.extend(preds.cpu().tolist())
            targets.extend(target.cpu().tolist())
            
    if evaluate:
        # Plot real targets vs predicted values
        plt.figure(figsize=(10, 5))
        plt.plot(targets, label="Real Target")
        plt.plot(predictions, label="Predicted")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.title("Real Target vs Predicted")
        plt.legend()
        plt.show()
    else:
        # Print predictions if not evaluating
        for idx, pred in enumerate(predictions):
            print(f"Sample {idx}: Predicted: {pred}")

def main():
    """
    Description:
        Parses command-line arguments and executes inference.
    Args:
        None
    Raises:
        Exception: If required command-line arguments are missing.
    Return:
        None
    """
    parser = argparse.ArgumentParser(description="Inference and Evaluation for Baseline Lighting Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_file", type=str, required=True, help="Path to data file for inference")
    parser.add_argument("--evaluate", action="store_true", help="If set, plots real target vs predicted values")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to run inference on")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    
    args = parser.parse_args()
    run_inference(args.checkpoint, args.data_file, args.evaluate, args.batch_size, args.num_samples)

if __name__ == "__main__":
    main()
