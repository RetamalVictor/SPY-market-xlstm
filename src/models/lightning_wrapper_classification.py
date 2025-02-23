import torch
import pytorch_lightning as pl
import torchmetrics

class GenericLightningModule(pl.LightningModule):
    """
    Description:
        A generic PyTorch Lightning module that wraps any torch.nn.Module.
        If scheduler parameters are provided, configures a cosine annealing scheduler with warmup.
    Args:
        model (torch.nn.Module): The underlying torch model.
        optimizer_class (torch.optim.Optimizer): The optimizer class to use.
        optimizer_kwargs (dict): Keyword arguments for the optimizer.
        loss_fn (callable): Loss function to use.
        warmup_steps (int, optional): Number of steps for linear warmup.
        total_steps (int, optional): Total number of training steps.
    Raises:
        None
    Return:
        An instance of GenericLightningModule.
    """
    def __init__(self, model: torch.nn.Module, optimizer_class=torch.optim.Adam,
                 optimizer_kwargs=None, loss_fn=None, warmup_steps: int = None, total_steps: int = None):
        super(GenericLightningModule, self).__init__()
        self.save_hyperparameters(ignore=['model', 'loss_fn', 'optimizer_class'])
        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.BCEWithLogitsLoss()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            Forward pass that delegates to the underlying torch model.
        Args:
            x (torch.Tensor): Input tensor.
        Return:
            torch.Tensor: Output of the underlying model.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        outputs, _ = self(inputs)
        preds = outputs[:, -1, :]

        loss = self.loss_fn(preds, targets.unsqueeze(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        prob = torch.sigmoid(preds)
        preds_binary = (prob > 0.5).float()
        acc = (preds_binary == targets.unsqueeze(-1)).float().mean()
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        inputs, targets = batch
        outputs, _ = self(inputs)            # outputs: [B, T, 1]
        preds = outputs[:, -1, :]           # shape [B, 1]

        loss = self.loss_fn(preds, targets.unsqueeze(-1))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        probs = torch.sigmoid(preds)             # [B, 1]
        preds_binary = (probs > 0.5).float()     # [B, 1]
        preds_binary = preds_binary.squeeze(-1)  # => [B]

        targets = targets.long()                 # => [B]

        # Now both preds_binary and targets have shape [B]
        acc = (preds_binary == targets).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # F1 score for binary classification (TorchMetrics >= 0.11)
        # Make sure to specify task="binary" in newer versions.
        f1 = torchmetrics.functional.f1_score(
            preds_binary,
            targets,
            task="binary"  # or 'binary' depending on your TorchMetrics version
        )
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "val_acc": acc, "val_f1": f1}



    def configure_optimizers(self):
        """
        Description:
            Configures the optimizer and, if warmup parameters are provided, a cosine scheduler with warmup.
        Return:
            Optimizer (and scheduler) configuration.
        """
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        if self.warmup_steps is not None and self.total_steps is not None and hasattr(self.model, "configure_scheduler"):
            scheduler = self.model.configure_scheduler(optimizer, self.warmup_steps, self.total_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }
        return optimizer
