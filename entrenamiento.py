import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modelo import LSTM
from montar_datos import TimeSeriesDataset

# Hyperparameters
input_size = 12
longitud_secuencia = 5
hidden_size = 1
output_size = 1
batch_size = 32
batch_size_val = 8
num_epochs = 15
learning_rate = 3e-4 # 0.0003

# Crear dataset -> dataloader
# __init__(self, train_ratio=0.75, validation_ratio=0.15, limite_capado=3, longitud_secuencia=5):

dataset = TimeSeriesDataset()
dataset.añadir_datos('dataset')

dataset.cambiar_modo('train')
entrenamiento_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataset.cambiar_modo('validation')
validation_dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=True)

# definir modelo
modelo = LSTM(
    input_size=12,
    hidden_size=1,
    output_size=1,
    longitud_secuencia=5
)

# Definir Loses
variable_para_nombrar_perdida = nn.MSELoss()
variable_para_nombrar_perdida_1 = ...

# Definir optimizador
optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)

perdidas_entrenamiento = []
perdidas_validation = []
for epoch in range(num_epochs):
    perdida_por_epoch = 0

    # entrenamiento
    for data_point in iter(entrenamiento_dataloader):
        # separar label y datos
        datos, label  = data_point
        # pasar datos por el modelo
        h0, c0 = modelo.init_hidden(batch_size=batch_size)
        try: 
            output, _ = modelo(datos.reshape(longitud_secuencia, batch_size, input_size))
            # calcular loss
            perdida_calculada = variable_para_nombrar_perdida(output, label.reshape(-1, 1))
            # guardar loss
            perdida_por_epoch += perdida_calculada.item()
            # backprop
            optimizer.zero_grad()
            perdida_calculada.backward()
            optimizer.step()
        except:
            continue
    
    print("Epoch: {}, loss: {}".format(epoch, perdida_por_epoch))
    perdidas_entrenamiento.append(perdida_por_epoch)
    
    # validacion
    perdida_por_epoch_validation = 0
    for data_point in iter(validation_dataloader):
        # separar label y datos
        datos, label  = data_point
        # pasar datos por el modelo
        with torch.no_grad():
            h0, c0 = modelo.init_hidden(batch_size=batch_size_val)
            try: 
                output, _ = modelo(datos.reshape(longitud_secuencia, batch_size_val, input_size))
                # calcular loss
                perdida_calculada_validation = variable_para_nombrar_perdida(output, label.reshape(-1, 1))
                # guardar loss
                perdida_por_epoch_validation += perdida_calculada.item()
            except:
                continue
    
    print("Epoch: {}, loss: {}".format(epoch, perdida_por_epoch_validation))
    perdidas_validation.append(perdida_por_epoch_validation)



# Plotear perdidas
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1)
ax[0].plot(perdidas_entrenamiento)
ax[1].plot(perdidas_validation)
plt.show()

# plt.plot(perdidas_entrenamiento)
# plt.plot(perdidas_validation)
# plt.show()

#  --> entrenamiento, validacion y test



    




