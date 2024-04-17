import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from custom_loss.directional import DirectionalAccuracyLoss
from montar_datos import TimeSeriesDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger

class LSTMModelParallelLightning(pl.LightningModule):
    def __init__(self, input_features=10, hidden_size=20, output_size=1, num_layers=1, dropout_rate=0.5, learning_rate=3e-4, alpha=0.75):
        super().__init__()
        self.alpha = alpha
        self.save_hyperparameters()
        # Define the model architecture
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(in_features=input_features, out_features=hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.activation = nn.ReLU()
        self.output_linear = nn.Linear(in_features=(3*hidden_size)+input_features, out_features=output_size)
        # Define the loss functions
        self.criterion_rmse = nn.MSELoss()
        self.criterion_dir_acc = DirectionalAccuracyLoss()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        mlp_output = self.linear(x[:, -1, :])
        mlp_output = self.batch_norm(mlp_output)
        mlp_output = self.activation(mlp_output)
        combined_output = torch.cat((lstm_out, mlp_output, x[:, -1, :]), dim=1)
        final_output = self.output_linear(combined_output)
        return final_output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        datos, label = batch
        y_pred = self(datos)
        loss_rmse = self.criterion_rmse(y_pred.reshape(-1, 1), label.reshape(-1, 1))
        loss_dir_acc = self.criterion_dir_acc(y_pred.reshape(-1, 1), label.reshape(-1, 1))
        train_loss = self.alpha * loss_rmse + (1-self.alpha) * loss_dir_acc
        self.log('train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        datos, label = batch
        y_pred = self(datos)
        loss_rmse = self.criterion_rmse(y_pred.reshape(-1, 1), label.reshape(-1, 1))
        loss_dir_acc = self.criterion_dir_acc(y_pred.reshape(-1, 1), label.reshape(-1, 1))
        val_loss = self.alpha * loss_rmse + (1-self.alpha) * loss_dir_acc
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss


if __name__ =="__main__":
    # Set up data loaders
    batch_size = batch_size_val = 32
    dataset = TimeSeriesDataset(longitud_secuencia=12)
    dataset.añadir_datos('dataset')


    dataset.cambiar_modo('train')
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=15, persistent_workers=True)
    for data_point in iter(train_dataloader):
        # separar label y datos
        datos, label  = data_point
        train_shape = datos.shape
        label_train_shape = label.shape
        break

    dataset.cambiar_modo('validation')
    val_dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=False, num_workers=15, persistent_workers=True)
    for data_point in iter(val_dataloader):
        # separar label y datos
        datos, label  = data_point
        val_shape = datos.shape
        label_val_shape = label.shape
        break

    # Set up the model and trainer
    model = LSTMModelParallelLightning(
        input_features=train_shape[-1],
        hidden_size=40,
        output_size=1, 
        num_layers=2, 
        learning_rate=1e-4
        )
    # Set up the logger
    logger = TensorBoardLogger("lightning_logs", name="my_model")

    # Training
    checkpoint_callback = ModelCheckpoint(
        dirpath=r"C:\Users\victo\Desktop\Clase_Walter\examples\ejemplo_LSTM_real\checkpoints",
        filename="{epoch}-{step}-{val_loss:.2f}",
        save_top_k=3,  # Saves the top 3 models
        verbose=True,
        monitor='val_loss',  # or 'val_accuracy' or another metric
        mode='min',  # or 'max' for metrics like accuracy
    )

    trainer = pl.Trainer(
        max_epochs=2500, 
        log_every_n_steps=5, 
        callbacks=[checkpoint_callback],
        logger=logger
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)
