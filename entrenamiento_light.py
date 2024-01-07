from montar_datos import TimeSeriesDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint


class MetricLoggingCallback(pl.Callback):
    def __init__(self):
        self.metrics = []

    def on_train_batch_end(self, trainer, *args, **kwargs):
        self.metrics.append(trainer.logged_metrics)

class LightningLSTM(pl.LightningModule):
    def __init__(self, num_features, hidden_size, learning_rate=0.001):
        super(LightningLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(num_features, hidden_size, batch_first=False)
        self.learning_rate = learning_rate
        self.longitud_seq = 5
        self.num_feat = 12
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        h0 = torch.zeros(1, x.size(1), self.hidden_size).cuda()
        c0 = torch.zeros(1, x.size(1), self.hidden_size).cuda()
        out, _ = self.lstm(x, (h0, c0))
        return out[-1, :, :]

    def training_step(self, batch, batch_idx):
        seq, target = batch
        seq = seq.reshape(self.longitud_seq, -1, self.num_feat)
        assert seq.shape[0] == self.longitud_seq, "La longitud de la secuencia no es la correcta, es {}".format(seq.shape[0])
        assert seq.shape[2] == self.num_feat, "El numero de features no es el correcto, es {}".format(seq.shape[2])
        output = self(seq)
        loss = nn.MSELoss()(output, target.reshape(-1, 1))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# Data Preparation (Assuming you have your dataset as tensors)

dataset = TimeSeriesDataset()
dataset.añadir_datos('dataset')
batch_size = 32
dataset.cambiar_modo('train')
entrenamiento_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model Initialization
model = LightningLSTM(num_features=12, hidden_size=1, learning_rate=0.001)

# Metrics
metric_logging_callback = MetricLoggingCallback()

# Training
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="{epoch}-{step}-{val_loss:.2f}",
    save_top_k=3,  # Saves the top 3 models
    verbose=True,
    monitor='val_loss',  # or 'val_accuracy' or another metric
    mode='min',  # or 'max' for metrics like accuracy
)

trainer = pl.Trainer(
    max_epochs=10,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback, metric_logging_callback]
)
trainer.fit(model, entrenamiento_dataloader)


# Assuming your trainer variable is named 'trainer'
metrics  = metric_logging_callback.metrics
metrics_np = np.array([[x['train_loss'].cpu().numpy()] for x in metrics])
# Plotting
plt.plot(metrics)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()