import torch
import torch.nn as nn
import pytorch_lightning as pl

class BaselineLightingModel(pl.LightningModule):
    """
    Description: Baseline model for lighting/trading data using an LSTM.
    Args:
        input_size (int): Number of input features per timestep.
        hidden_size (int): Hidden size for the LSTM.
        num_layers (int): Number of LSTM layers.
        learning_rate (float): Learning rate for the optimizer.
    Raises:
        ValueError: If input_size, hidden_size, or num_layers are not positive integers.
    Return:
        A scalar prediction for each input sequence.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, learning_rate: float):
        super(BaselineLightingModel, self).__init__()
        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("input_size, hidden_size, and num_layers must be positive integers")
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description: Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        Return:
            torch.Tensor: Output tensor of shape (batch_size).
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Description: Training step for the model.
        Args:
            batch: Tuple containing (inputs, targets).
            batch_idx (int): Index of the batch.
        Return:
            torch.Tensor: Loss value.
        """
        inputs, targets = batch
        preds = self.forward(inputs)
        loss = self.criterion(preds, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        """
        Description: Validation step for the model.
        Args:
            batch: Tuple containing (inputs, targets).
            batch_idx (int): Index of the batch.
        """
        inputs, targets = batch
        preds = self.forward(inputs)
        loss = self.criterion(preds, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Description: Configures the optimizer.
        Return:
            torch.optim.Optimizer: The optimizer for the model.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
