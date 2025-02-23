#!/usr/bin/env python
import argparse
import os
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: scikit-learn for metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

# Project modules
from src.dataset_spy import TradingDataset
from src.config import get_model_class, load_config
from src.models.lightning_wrapper_classification import GenericLightningModule

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
            # If classification, use BCEWithLogitsLoss; otherwise, MSELoss or another regression loss
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
    
    # 6. Run inference
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, target_batch in test_loader:
            inputs = inputs.to(device)
            target_batch = target_batch.to(device)

            preds, _ = model(inputs)    # shape: [B, seq, output_dim]
            preds = preds[:, -1, :]     # final time step => shape [B, output_dim]

            if classification:
                # Convert logits -> probabilities -> binary predictions
                probs = torch.sigmoid(preds.squeeze(-1))  # shape [B]
                preds_binary = (probs > 0.5).float()
                predictions.extend(preds_binary.cpu().tolist())
            else:
                # For regression, just use raw predictions
                predictions.extend(preds.squeeze(-1).cpu().tolist())

            # Collect targets
            targets.extend(target_batch.cpu().tolist())
    
    # 7. Evaluate or print results
    if evaluate:
        plt.figure(figsize=(10, 5))
        plt.plot(targets, label="Real Target")
        plt.plot(predictions, label="Predicted")
        plt.xlabel("Sample Index")
        plt.ylabel("Value" if not classification else "Class")
        plt.title("Real vs. Predicted Targets")
        plt.legend()
        plt.savefig("inference_results.png", dpi=150)
        plt.close()
        print("Inference plot saved as 'inference_results.png'.")
    else:
        # Print predictions vs. targets
        for idx, pred in enumerate(predictions):
            print(f"Sample {idx}: Predicted: {pred}, Real: {targets[idx]}")

    # 8. Compute metrics
    if classification:
        # Convert to int in case we have floats
        predictions_int = [int(p) for p in predictions]
        targets_int = [int(t) for t in targets]

        # Accuracy, precision, recall, F1
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(targets_int, predictions_int)
        prec = precision_score(targets_int, predictions_int, average='binary', zero_division=0)
        rec = recall_score(targets_int, predictions_int, average='binary', zero_division=0)
        f1 = f1_score(targets_int, predictions_int, average='binary', zero_division=0)
        print("\nClassification Metrics (Binary):")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
    else:
        # Example regression metric: MSE
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(targets, predictions)
        print(f"\nRegression Metric:")
        print(f"  MSE: {mse:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for a trained model using a preprocessed test file from a directory specified in the config."
    )
    parser.add_argument("--checkpoint", type=str, required=False, default=r"tb_logs\spy_xlstm_clas\version_5\checkpoints\best-epoch=10-val_loss=0.69.ckpt",
                        help="Path to the trained model checkpoint. If not provided, a new model will be created.")
    parser.add_argument("--config", type=str, required=False, default="configs/train_config.yaml",
                        help="Path to YAML configuration file (e.g., train_config.yaml)")
    parser.add_argument("--evaluate", action="store_true",
                        help="If set, plots predictions vs. real targets; otherwise, prints predictions.")
    parser.add_argument("--classification", action="store_true",
                        help="If set, treats the model output as binary classification (logits). Otherwise, uses regression logic.")
    parser.add_argument("--num_samples", type=int, default=900,
                        help="Number of samples to run inference on (default: 900)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference (default: 64)")
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
