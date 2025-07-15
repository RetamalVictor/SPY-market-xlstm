import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl

class InferencePlotCallback(pl.Callback):
    """
    Lightning callback that runs inference on a provided validation DataLoader after
    each validation epoch and logs plots to TensorBoard.

    In multi-task mode (pl_module.multi_task == True), it creates three subplots:
      1. Predicted Sign vs. Real Sign.
      2. Predicted Normalized Spy Value vs. Real Normalized Spy Value.
      3. Final Real Value (sign * denormalized spy) vs. True Real Value.

    In single-task mode, it plots predicted targets vs. real targets in a single subplot.

    Args:
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        num_batches (int): Number of batches to run inference on (default: 100).
    """
    def __init__(self, val_dataloader, num_batches: int = 100):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.num_batches = num_batches

        # Load spy normalization parameters once from the dataset.
        dataset = self.val_dataloader.dataset
        # If dataset is a Subset, retrieve the underlying dataset.
        if hasattr(dataset, "norm_params"):
            norm_params = dataset.norm_params
        elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "norm_params"):
            norm_params = dataset.dataset.norm_params
        else:
            norm_params = None

        if norm_params is not None and "spy" in norm_params:
            self.spy_min = norm_params["spy"]["min"]
            self.spy_max = norm_params["spy"]["max"]
        else:
            self.spy_min, self.spy_max = 0.0, 1.0  # Fallback defaults

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        device = pl_module.device

        # Initialize lists for accumulating values.
        predicted_sign_list = []
        true_sign_list = []
        predicted_norm_value_list = []
        true_norm_value_list = []
        final_pred_list = []
        final_true_list = []

        batch_count = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, target_batch = batch
                inputs = inputs.to(device)
                target_batch = target_batch.to(device)
                preds = pl_module(inputs)
                if isinstance(preds, tuple):
                    preds = preds[0]
                # Assume preds shape is [B, T, output_dim]; take last time step.
                last_preds = preds[:, -1, :]  # shape: [B, output_dim]

                # Extract sign and normalized spy value.
                sign_logits = last_preds[:, 0]      # [B]
                spy_pred_norm = last_preds[:, 1]      # [B]

                # Process predicted sign: sigmoid -> threshold -> map to ±1.
                sign_prob = torch.sigmoid(sign_logits)
                sign_binary = (sign_prob > 0.5).float()  # 0 or 1
                predicted_sign = sign_binary * 2 - 1       # -1 or +1

                # Raw predicted normalized spy value.
                predicted_norm_value = spy_pred_norm

                # Denormalize spy prediction.
                spy_denorm = ((spy_pred_norm + 1) / 2) * (self.spy_max - self.spy_min) + self.spy_min
                final_pred = predicted_sign * spy_denorm

                # Process targets (expected shape [B, 2]):
                target_sign = target_batch[:, 0]       # stored as 0/1
                true_sign = target_sign * 2 - 1          # convert to -1/+1
                true_norm_value = target_batch[:, 1]
                spy_target_denorm = ((true_norm_value + 1) / 2) * (self.spy_max - self.spy_min) + self.spy_min
                final_true = true_sign * spy_target_denorm

                predicted_sign_list.extend(predicted_sign.cpu().tolist())
                true_sign_list.extend(true_sign.cpu().tolist())
                predicted_norm_value_list.extend(predicted_norm_value.cpu().tolist())
                true_norm_value_list.extend(true_norm_value.cpu().tolist())
                final_pred_list.extend(final_pred.cpu().tolist())
                final_true_list.extend(final_true.cpu().tolist())

                batch_count += 1
                if batch_count >= self.num_batches:
                    break

        # Plotting.
        x_axis = range(len(true_sign_list))
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        # Subplot 1: Predicted Sign vs. Real Sign.
        axes[0].plot(x_axis, true_sign_list, label="Real Sign", marker='o', linestyle='-')
        axes[0].plot(x_axis, predicted_sign_list, label="Predicted Sign", marker='x', linestyle='--')
        axes[0].set_ylabel("Sign (-1 or +1)")
        axes[0].set_title("Predicted Sign vs. Real Sign")
        axes[0].legend()

        # Subplot 2: Predicted Normalized Spy vs. Real Normalized Spy.
        axes[1].plot(x_axis, true_norm_value_list, label="Real Norm Spy", marker='o', linestyle='-')
        axes[1].plot(x_axis, predicted_norm_value_list, label="Predicted Norm Spy", marker='x', linestyle='--')
        axes[1].set_ylabel("Normalized Spy Value")
        axes[1].set_title("Predicted Normalized Spy vs. Real")
        axes[1].legend()

        # Subplot 3: Final Real Value vs. True Real Value.
        axes[2].plot(x_axis, final_true_list, label="Real Final Value", marker='o', linestyle='-')
        axes[2].plot(x_axis, final_pred_list, label="Predicted Final Value", marker='x', linestyle='--')
        axes[2].set_xlabel("Sample Index")
        axes[2].set_ylabel("Final Value")
        axes[2].set_title("Final Real Value vs. Predicted")
        axes[2].legend()

        if trainer.logger is not None:
            trainer.logger.experiment.add_figure(
                "Inference/Real_vs_Predicted_Detailed", fig, global_step=trainer.global_step
            )
        plt.close(fig)
        pl_module.train()
