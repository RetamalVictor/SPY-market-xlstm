import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


"""
No te preocupes ahora de los datos. Lo unico que tienes que saber son las dimensiones
Recuerda, [longitud_sequ, muestra, numer_feat] [12, N, 8].
En este caso para no complicarnos vamos a entrenar con todos los datos cada epoch.

Nos centramos solo en modelar la arquitectura.
"""
# Por ahora vamos a usar solo la cpu
device = torch.device("cpu")

# Load the dataset
df = pd.read_csv(r'examples\Ejemplo_LSTM\time_series_data.csv')

# Select features and target
features = df.drop(columns=['x', 'combined_series_correlated_disturbed'])
target = df['combined_series_correlated_disturbed']

# Aqui una normalizacion para poner ejemplo
scaler_features = MinMaxScaler(feature_range=(-1, 1))
scaler_target = MinMaxScaler(feature_range=(-1, 1))
features_normalized = scaler_features.fit_transform(features)
target_normalized = scaler_target.fit_transform(target.values.reshape(-1, 1))

# Los hacemos tensores
features_normalized = torch.FloatTensor(features_normalized).to(device)
target_normalized = torch.FloatTensor(target_normalized).view(-1).to(device)

# # creamos las secuencias
# def create_inout_sequences(input_features, input_target, tw):
#     inout_seq = []
#     L = len(input_target)
#     for i in range(L - tw):
#         train_seq = input_features[i:i + tw]
#         train_label = input_target[i + tw:i + tw + 1]
#         inout_seq.append([train_seq, train_label])
#     return inout_seq

# # Create input-output sequence pairs
# train_window = 12
# train_inout_seq = create_inout_sequences(features_normalized[:-200], target_normalized[:-200], train_window)
# val_inout_seq = create_inout_sequences(features_normalized[-200:], target_normalized[-200:], train_window)

def create_inout_sequences(input_features, input_target, tw, batch_size=None, shuffle=True):
    inout_seq = []
    if batch_size is None:
        batch_size=1
    L = len(input_target)
    for i in range(L - tw):
        train_seq = input_features[i:i + tw]
        train_label = input_target[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    # Convertimos las secuencias en tensores y las cargamos en un DataLoader para manejar los minibatches
    inout_seq = [torch.stack(tensors) for tensors in zip(*inout_seq)]
    inout_seq = TensorDataset(*inout_seq)
    inout_loader = DataLoader(inout_seq, batch_size=batch_size, shuffle=shuffle)
    return inout_loader

# Tamaño de la ventana de entrenamiento y del batch
train_window = 12
batch_size = 32

# Creamos secuencias de entrenamiento y validación
train_inout_seq = create_inout_sequences(features_normalized[:-200], target_normalized[:-200], train_window, batch_size)
val_inout_seq = create_inout_sequences(features_normalized[-200:], target_normalized[-200:], train_window, batch_size)
for seq, labels in train_inout_seq:
        seq, labels = seq.to(device), labels.to(device)
        print(f"Las dimensiones de entrenamiento son: \nBatch size: {seq.shape[0]}, seq_lenght: {seq.shape[1]}, input_ft: {seq.shape[2]}")
        print(f"La labels:Batch size: {labels.shape[0]}, label_ft: {labels.shape[1]}")
        break

#### Aqui empieza lo bueno ###
"""
La idea es la siguiente. Hay un modelo LSTM creado. 
Tu tienes que crear el segundo modelo lstm con la arquitectura en paralelo como en el diagrama

Pistas:
- Cada bloque del diagrama es una linea tipo: 
            self.lstm = nn.LSTM(...) o self.linear = nn.Linear(...)
- Cuando van en serie, el output de una entra en la siguiente
            lstm_out, _ = self.lstm(x)
            output = self.linear(lstm_out[-1])
- cuando van en paralelo, tienes que hacer una operacion de concatenacion extra
            # ouput del bloque 1 del paralelo       # Output del bloque 2 del paralelo
            lstm_out, _ = self.lstm(x)                  mlp_output = self.linear(x_flattened)  <-- Aqui tienes que decidir si aplanas toda la sequencia o usas solo el ultimo dia
                                                                                                 x_flattened = x.view(x.shape[1], -1)       ;           x_flattened = x[-1, :, :]
    #   No te olvides de seleccionar la parte de la prediccion final que es relevante para ti
                lstm_out[-1, :, :]                        
                                # Convinacion de los dos outputs
                    combined_output = torch.cat((lstm_out, mlp_output), dim=1)

Consejo:
- Utiliza algun metodo para clarificar las dimensiones de cada parte primero. 
"""
    
class LSTMModel_parallel(nn.Module):
    """
    Esta version utiliza el ultimo dia de cada trozo de la serie
    
    """
    def __init__(self, input_features=10, hidden_size=20, output_size=1, num_layers=1):
        super(LSTMModel_parallel, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        # Linear layer en paralelo al LSTM
        self.linear = nn.Linear(in_features=input_features, out_features=hidden_size)
        # Capa lineal que recibe la concatenación de las salidas de LSTM y la capa lineal
        self.output_linear = nn.Linear(in_features=2*hidden_size, out_features=output_size)

    def forward(self, x):
        # x debe tener la forma (batch_size, seq_lenght, input_features)
        # Salida del LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out tiene la forma (batch_size, seq_lenght, hidden_size)
        # Tomamos solo la última salida del LSTM para cada secuencia
        lstm_out = lstm_out[:, -1, :]  # Ahora lstm_out tiene la forma (batch_size, hidden_size)

        # Salida de la capa lineal
        #opcion 1: solo el ultimo dia de la sequencia
        mlp_output = self.linear(x[:, -1, :])  # Usamos solo el último elemento de la secuencia
        
        # Concatenamos las salidas del LSTM y la capa lineal
        combined_output = torch.cat((lstm_out, mlp_output), dim=1)

        # Pasamos la salida combinada a través de la capa lineal final
        final_output = self.output_linear(combined_output)

        return final_output
# Set hyperparameters
input_size = features.shape[1]
hidden_size = 40
output_size = 1
num_layers = 2
learning_rate = 3e-4
num_epochs = 10

# Instantiate the models
lstm_model = LSTMModel_parallel(input_size, hidden_size, output_size, num_layers).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=learning_rate)

# Function to evaluate the model on validation data
def evaluate_model(model, val_seq):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for seq, labels in val_seq:
            seq, labels = seq.to(device), labels.to(device)
            y_pred = model(seq)
            loss = criterion(y_pred.reshape(-1,1), labels.reshape(-1,1))
            val_loss += loss.item()
        return val_loss / len(val_seq)

# Bucle de entrenamiento para el modelo LSTM
print("Entrenando el modelo LSTM en paralelo...")
train_losses_lstm = []
val_losses_lstm = []
for epoch in range(num_epochs):
    lstm_model.train()
    model_loss = 0.0
    for seq, labels in train_inout_seq:
        seq, labels = seq.to(device), labels.to(device)  # Aseguramos que los datos estén en el dispositivo correcto
        optimizer_lstm.zero_grad()
        y_pred = lstm_model(seq)
        loss = criterion(y_pred.reshape(-1, 1), labels.reshape(-1, 1))
        loss.backward()
        optimizer_lstm.step()
        model_loss += loss.item()

    train_loss = model_loss / len(train_inout_seq.dataset)
    train_losses_lstm.append(train_loss)
    val_loss = evaluate_model(lstm_model, val_inout_seq)
    val_losses_lstm.append(val_loss)

    if epoch % 3 == 0:
        print(f'Epoch {epoch} Train Loss {train_loss} Val Loss {val_loss}')


# Function to evaluate the model on validation data
def predict(model, val_seq):
    model.eval()
    target = []
    predictions =[]
    with torch.no_grad():
        for seq, labels in val_seq:
            seq, labels = seq.to(device), labels.to(device)
            y_pred = model(seq)
            predictions.append(y_pred)
            target.append(labels)
        return predictions, target


inout_seq = create_inout_sequences(features_normalized, target_normalized, train_window, shuffle=False)

predictions_lstm ,targets_lstm=predict(lstm_model, inout_seq)
# Assuming predictions_lstm is a list of tensors
predictions_lstm = torch.cat(predictions_lstm, dim=0)
predictions_lstm = predictions_lstm.reshape(-1).cpu().numpy()

# Assuming targets_lstm is a tensor
targets_lstm = torch.cat(targets_lstm, dim=0)
targets_lstm = targets_lstm.reshape(-1).cpu().numpy()
# Aqui empezamos a crear los graficos. 
# Setting up a 1x2 subplot grid
fig, axs = plt.subplots(2, 1, figsize=(18, 12))

# Plotting targets vs predictions for LSTM
axs[0].plot(targets_lstm, label='Ground Truth (LSTM)')
axs[0].plot(predictions_lstm, label='LSTM Predictions')
axs[0].set_title('Ground Truth vs LSTM Predictions')
axs[0].set_xlabel('Time Step')
axs[0].set_ylabel('Normalized Value')
axs[0].legend()

# Plotting training vs validation loss for LSTM
axs[1].plot(train_losses_lstm, label='LSTM Train Loss')
axs[1].plot(val_losses_lstm, label='LSTM Val Loss')
axs[1].set_title('LSTM Training vs Validation Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()