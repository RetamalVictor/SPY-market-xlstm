import torch
import pytorch_lightning as pl
import torchmetrics

class GenericLightningModule(pl.LightningModule):
    """
    Description:
        A generic PyTorch Lightning module that wraps any torch.nn.Module.
        Supports both single-task and multi-task (double-task) training.
    Args:
        model (torch.nn.Module): The underlying torch model.
        optimizer_class (torch.optim.Optimizer): The optimizer class to use.
        optimizer_kwargs (dict): Keyword arguments for the optimizer.
        loss_fn (callable): Loss function for single-task mode.
        warmup_steps (int, optional): Number of steps for linear warmup.
        total_steps (int, optional): Total number of training steps.
    Returns:
        An instance of GenericLightningModule.
    """
    def __init__(self, model: torch.nn.Module, optimizer_class=torch.optim.Adam,
                optimizer_kwargs=None, loss_fn=None, warmup_steps: int = None, total_steps: int = None):
        super(GenericLightningModule, self).__init__()
        self.save_hyperparameters(ignore=['model', 'loss_fn', 'optimizer_class'])
        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        # Always define multi-task losses so they exist even when not used.
        self.loss_class_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_regr_fn = torch.nn.MSELoss()
        if getattr(self.hparams, "output_size", 1) == 2:
            self.multi_task = True
        else:
            self.multi_task = False
            self.loss_fn = loss_fn if loss_fn is not None else torch.nn.BCEWithLogitsLoss()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            Forward pass that delegates to the underlying torch model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output from the underlying model.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Description:
            Computes the training loss and logs relevant metrics.
        Args:
            batch: Batch of data.
            batch_idx (int): Batch index.
        Returns:
            torch.Tensor: Computed training loss.
        """
        inputs, targets = batch
        outputs, _ = self(inputs)
        preds = outputs[:, -1, :]  # shape: [B, output_size]
        if self.multi_task:
            class_pred = preds[:, 0]
            regr_pred = preds[:, 1]
            sign_target = targets[:, 0]
            magnitude_target = targets[:, 1]
            bce_loss = self.loss_class_fn(class_pred, sign_target)
            mse_loss = self.loss_regr_fn(regr_pred, magnitude_target)
            loss = bce_loss + mse_loss
            prob = torch.sigmoid(class_pred)
            preds_binary = (prob > 0.5).float()
            acc = (preds_binary == sign_target).float().mean()
            mae = torch.mean(torch.abs(regr_pred - magnitude_target))
            self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_mae", mae, on_step=True, on_epoch=True, prog_bar=True)
        else:
            loss = self.loss_fn(preds, targets.unsqueeze(-1))
            prob = torch.sigmoid(preds)
            preds_binary = (prob > 0.5).float()
            acc = (preds_binary == targets.unsqueeze(-1)).float().mean()
            self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        """
        Description:
            Computes the validation loss and logs relevant metrics.
        Args:
            batch: Batch of data.
            batch_idx (int): Batch index.
        Returns:
            dict: Dictionary containing validation loss and metrics.
        """
        inputs, targets = batch
        outputs, _ = self(inputs)
        preds = outputs[:, -1, :]  # shape: [B, output_dim]
        
        # If targets have two elements per sample, use the multi-task branch.
        if (targets.ndim > 1 and targets.shape[1] == 2) or self.multi_task:
            class_pred = preds[:, 0]
            regr_pred = preds[:, 1]
            sign_target = targets[:, 0]
            magnitude_target = targets[:, 1]
            bce_loss = self.loss_class_fn(class_pred, sign_target)
            mse_loss = self.loss_regr_fn(regr_pred, magnitude_target)
            loss = bce_loss + mse_loss
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            prob = torch.sigmoid(class_pred)
            preds_binary = (prob > 0.5).float()
            acc = (preds_binary == sign_target).float().mean()
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            f1 = torchmetrics.functional.f1_score(preds_binary, sign_target, task="binary")
            self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True)
            mae = torch.mean(torch.abs(regr_pred - magnitude_target))
            self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
            return {"val_loss": loss, "val_acc": acc, "val_f1": f1, "val_mae": mae}
        else:
            # Single-task branch: unsqueeze targets as they are scalars.
            loss = self.loss_fn(preds, targets.unsqueeze(-1))
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            prob = torch.sigmoid(preds)
            preds_binary = (prob > 0.5).float().squeeze(-1)
            targets_int = targets.long()
            acc = (preds_binary == targets_int).float().mean()
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            f1 = torchmetrics.functional.f1_score(preds_binary, targets_int, task="binary")
            self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True)
            return {"val_loss": loss, "val_acc": acc, "val_f1": f1}

    def configure_optimizers(self):
        """
        Description:
            Configures the optimizer and cosine scheduler with warmup if parameters are provided.
        Returns:
            dict or torch.optim.Optimizer: Optimizer (and scheduler) configuration.
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
