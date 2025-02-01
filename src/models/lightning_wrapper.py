import torch
import pytorch_lightning as pl

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
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss()
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
        """
        Description:
            Training step: computes predictions and loss.
        Args:
            batch: A batch of data (inputs, targets).
            batch_idx (int): Index of the batch.
        Return:
            torch.Tensor: The computed loss.
        """
        inputs, targets = batch
        preds = self(inputs).squeeze(-1)
        loss = self.loss_fn(preds, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        """
        Description:
            Validation step: computes predictions and loss.
        Args:
            batch: A batch of data (inputs, targets).
            batch_idx (int): Index of the batch.
        Return:
            dict: A dictionary containing predictions and targets.
        """
        inputs, targets = batch
        preds = self(inputs).squeeze(-1)
        loss = self.loss_fn(preds, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"preds": preds, "targets": targets}

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
