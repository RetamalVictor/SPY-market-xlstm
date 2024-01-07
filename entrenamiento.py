import numpy as np

import torch
from torch.utils.data import DataLoader

from modelo import LSTM
from montar_datos import TimeSeriesDataset

# Hyperparameters
input_size = ...
longitud_secuencia = ...
hidden_size = ...
output_size = ...
batch_size = 32
num_epochs = ...
learning_rate = ...

# Crear dataset -> dataloader
# __init__(self, train_ratio=0.75, validation_ratio=0.15, limite_capado=3, longitud_secuencia=5):

dataset = TimeSeriesDataset()
dataset.añadir_datos('dataset')

dataset.cambiar_modo('train')
entrenamiento_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# definir modelo
modelo = LSTM(
    input_size=12,
    hidden_size=1,
    output_size=1,
    longitud_secuencia=5
)

for data_point in iter(entrenamiento_dataloader):
    # separar label y datos
    datos, label  = data_point
    # pasar datos por el modelo
    h0, c0 = modelo.init_hidden(batch_size=batch_size)
    output = modelo(datos)
    break
    # calcular loss
    # backprop

    




#  --> entrenamiento, validacion y test
# Inicializar el modelo
# Pasar datos por el modelo.



