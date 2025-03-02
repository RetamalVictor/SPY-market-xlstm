#!/usr/bin/env python
import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset_spy import TradingDataset
from src.config import get_model_class, load_config
from src.models.lightning_wrapper_multitask import GenericLightningModule

def run_inference(checkpoint: str,
                  config_path: str,
                  evaluate: bool,
                  classification: bool,
                  batch_size: int = 10,
                  num_samples: int = 100):
    """
    Description:
        Loads configuration from YAML, locates the test file in the given data directory,
        instantiates a test dataset (using the new TradingDataset) and DataLoader,
        and loads a trained model checkpoint if available. If no checkpoint exists, a new model is created.
        Inference is then run on the test dataset. If evaluate is set, the predictions are plotted against the real targets.
        This script can handle both regression and binary classification (controlled by `classification`).
    
    Args:
        checkpoint (str): Path to the trained model checkpoint.
        config_path (str): Path to the YAML configuration file (e.g., train_config.yaml).
        evaluate (bool): If True, plots predictions vs. real targets; otherwise, prints them.
        classification (bool): If True, treat model outputs as logits for binary classification.
        batch_size (int, optional): Batch size for inference.
        num_samples (int, optional): Number of samples to run inference on.
    
    Returns:
        None
    """
    # 1. Load configuration.
    dataset_config, training_config, model_config = load_config(config_path)
    dataset_config.mode = "test"

    if getattr(dataset_config, "multi_task", False):
        dataset_config.output_size = 2

    # 2. Load the test dataset.
    test_dataset = TradingDataset(dataset_config)
    print("Test dataset valid samples:", len(test_dataset))
    
    # Optionally restrict to a subset of samples.
    if num_samples < len(test_dataset):
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, list(range(num_samples)))
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 3. Determine input feature dimension dynamically.
    sample, _ = test_dataset[0]
    input_size = sample.shape[-1]
    print("Computed input feature dimension for model:", input_size)
    
    # 4. Build the model architecture from config.
    model_cls = get_model_class(training_config.model_name)
    torch_model = model_cls.from_config(input_size, training_config.output_size, model_config)
    
    # 5. Load the Lightning checkpoint if it exists.
    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading model from checkpoint: {checkpoint}")
        model = GenericLightningModule.load_from_checkpoint(
            checkpoint_path=checkpoint,
            model=torch_model,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={'lr': training_config.learning_rate},
            loss_fn=torch.nn.BCEWithLogitsLoss() if classification else torch.nn.MSELoss(),
            warmup_steps=training_config.warmup_steps,
            total_steps=training_config.total_steps
        )
    else:
        print("No valid checkpoint provided; creating a new model instance.")
        model = GenericLightningModule(
            model=torch_model,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={'lr': training_config.learning_rate},
            loss_fn=torch.nn.BCEWithLogitsLoss() if classification else torch.nn.MSELoss(),
            warmup_steps=training_config.warmup_steps,
            total_steps=training_config.total_steps
        )
    model.eval()
    device = next(model.parameters()).device
    model.to(device)
    
    # 6. Load norm_params for spy from the norm_param_path
    if getattr(dataset_config, "normalize", True):
        if not os.path.exists(dataset_config.norm_param_path):
            raise ValueError(f"Normalization parameters file {dataset_config.norm_param_path} not found.")
        with open(dataset_config.norm_param_path, "r") as f:
            norm_params = json.load(f)
        spy_min = norm_params["spy"]["min"]
        spy_max = norm_params["spy"]["max"]
    else:
        spy_min, spy_max = None, None  # Not used if normalization is off
    
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, target_batch in test_loader:
            inputs = inputs.to(device)
            target_batch = target_batch.to(device)  # shape: [B, 2] if multi-task
            
            preds, _ = model(inputs)    # shape: [B, seq, output_dim]
            preds = preds[:, -1, :]     # final time step, shape: [B, output_dim]
            
            if classification:
                # Classification branch (unchanged)
                probs = torch.sigmoid(preds.squeeze(-1))  # shape [B]
                preds_binary = (probs > 0.5).float()
                predictions.extend(preds_binary.cpu().tolist())
                targets.extend(target_batch.squeeze(-1).cpu().tolist())
            else:
                # Multi-task regression branch: convert predicted sign & normalized spy value
                # Process predictions:
                sign_logits = preds[:, 0]
                sign_prob = torch.sigmoid(sign_logits)
                sign_binary = (sign_prob > 0.5).float()  # 0 or 1
                # Convert to ±1: 0 -> -1, 1 -> +1
                sign_mult = sign_binary * 2 - 1  
                spy_pred_norm = preds[:, 1]
                # Denormalize the spy prediction
                spy_denorm = ((spy_pred_norm + 1) / 2) * (spy_max - spy_min) + spy_min
                final_pred = sign_mult * spy_denorm  # final predicted value
                predictions.extend(final_pred.cpu().tolist())
                
                # Process targets similarly:
                target_sign = target_batch[:, 0]
                target_sign_mult = target_sign * 2 - 1
                target_spy_norm = target_batch[:, 1]
                target_spy_denorm = ((target_spy_norm + 1) / 2) * (spy_max - spy_min) + spy_min
                final_target = target_sign_mult * target_spy_denorm
                targets.extend(final_target.cpu().tolist())
    
    # 7. Evaluate or print results
    if evaluate:
        plt.figure(figsize=(10, 5))
        plt.plot(targets, label="Real Target")
        plt.plot(predictions, label="Predicted")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.title("Real vs. Predicted Targets (Denormalized)")
        plt.legend()
        plt.savefig("inference_results.png", dpi=150)
        plt.close()
        print("Inference plot saved as 'inference_results.png'.")
    else:
        for idx, (pred, real) in enumerate(zip(predictions, targets)):
            print(f"Sample {idx}: Predicted: {pred}, Real: {real}")
    
    # 8. Compute and print a regression metric (e.g., MSE)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(targets, predictions)
    print(f"\nRegression Metric (MSE): {mse:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for a trained model using a preprocessed test file from a directory specified in the config."
    )
    parser.add_argument("--checkpoint", type=str, required=False, default="path/to/checkpoint.ckpt",
                        help="Path to the trained model checkpoint. If not provided, a new model will be created.")
    parser.add_argument("--config", type=str, required=False, default="configs/train_config.yaml",
                        help="Path to YAML configuration file (e.g., train_config.yaml)")
    parser.add_argument("--evaluate", action="store_true",
                        help="If set, plots predictions vs. real targets; otherwise, prints predictions.")
    parser.add_argument("--classification", action="store_true",
                        help="If set, treats the model output as binary classification (logits). Otherwise, uses regression logic.")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to run inference on.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference.")
    args = parser.parse_args()

    run_inference(
        checkpoint=args.checkpoint,
        config_path=args.config,
        evaluate=args.evaluate,
        classification=args.classification,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )

if __name__ == "__main__":
    main()
