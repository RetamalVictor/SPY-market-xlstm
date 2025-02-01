import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl

class InferencePlotCallback(pl.Callback):
    """
    Description:
        Lightning callback that runs inference on a provided validation DataLoader after
        each validation epoch and logs a plot comparing predicted vs. real targets to TensorBoard.
    Args:
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        num_batches (int): Number of batches to run inference on (default: 1).
    Raises:
        None
    Return:
        None
    """
    def __init__(self, val_dataloader, num_batches: int = 1):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.num_batches = num_batches

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Description:
            After validation ends, run inference on a subset of validation data and log a
            plot of predicted vs. real targets to TensorBoard.
        Args:
            trainer: The PyTorch Lightning Trainer.
            pl_module: The LightningModule being trained.
        Return:
            None
        """
        pl_module.eval()
        device = pl_module.device
        predictions = []
        targets = []
        batch_count = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, target = batch
                inputs = inputs.to(device)
                preds = pl_module(inputs)
                predictions.extend(preds.cpu().tolist())
                targets.extend(target.cpu().tolist())
                batch_count += 1
                if batch_count >= self.num_batches:
                    break

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(targets, label="Real Target")
        ax.plot(predictions, label="Predicted")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        ax.set_title("Real vs. Predicted")
        ax.legend()

        # Log the figure to TensorBoard
        if trainer.logger is not None:
            # trainer.logger.experiment is the TensorBoard SummaryWriter
            trainer.logger.experiment.add_figure(
                "Inference/Real_vs_Predicted", fig, global_step=trainer.global_step
            )
        plt.close(fig)
        pl_module.train()
