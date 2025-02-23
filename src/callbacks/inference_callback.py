import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl

class InferencePlotCallback(pl.Callback):
    """
    Lightning callback that runs inference on a provided validation DataLoader after
    each validation epoch and logs a plot comparing predicted vs. real targets to TensorBoard.
    Args:
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        num_batches (int): Number of batches to run inference on (default: 1).
    """
    def __init__(self, val_dataloader, num_batches: int = 100):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.num_batches = num_batches

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        After validation ends, run inference on a subset of validation data and log a
        plot of predicted vs. real targets (using only the last time step predictions)
        to TensorBoard.
        """
        pl_module.eval()
        device = pl_module.device
        predictions = []
        targets = []
        batch_count = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, target_batch = batch
                inputs = inputs.to(device)
                target_batch = target_batch.to(device)
                preds = pl_module(inputs)
                # If preds is a tuple, take the first element (the predictions).
                if isinstance(preds, tuple):
                    preds = preds[0]
                # Assume preds shape is [B, T, output_dim] and we want the last time step.
                last_preds = preds[:, -1, :].squeeze(-1)
                predictions.extend(last_preds.cpu().tolist())
                targets.extend(target_batch.cpu().tolist())
                batch_count += 1
                if batch_count >= self.num_batches:
                    break

        # Now plot the collected predictions vs. targets for all samples.
        if predictions and targets:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(len(targets)), targets, label="Real Target", marker='o')
            ax.plot(range(len(predictions)), predictions, label="Predicted", marker='x')
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Value")
            ax.set_title("Real vs. Predicted Targets")
            ax.legend()

            if trainer.logger is not None:
                # trainer.logger.experiment is the TensorBoard SummaryWriter.
                trainer.logger.experiment.add_figure(
                    "Inference/Real_vs_Predicted", fig, global_step=trainer.global_step
                )
            plt.close(fig)

        pl_module.train()
