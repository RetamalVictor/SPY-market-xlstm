import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from custom_loss.directional import DirectionalAccuracyLoss
from montar_datos import TimeSeriesDataset

# Por ahora vamos a usar solo la cpu
device = torch.device("cpu")
# Tamaño de la ventana de entrenamiento y del batch
batch_size = 28
batch_size_val = 28

dataset = TimeSeriesDataset(longitud_secuencia=12)
dataset.añadir_datos('dataset')

dataset.cambiar_modo('train')
entrenamiento_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(len(dataset))
# entrenamiento
for data_point in iter(entrenamiento_dataloader):
    # separar label y datos
    datos, label  = data_point
    train_shape = datos.shape
    label_train_shape = label.shape
    break

dataset.cambiar_modo('validation')
validation_dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=True)

# entrenamiento
for data_point in iter(validation_dataloader):
    # separar label y datos
    datos, label  = data_point
    val_shape = datos.shape
    label_val_shape = label.shape
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
    
# class LSTMModel_parallel(nn.Module):
#     """
#     Esta version utiliza el ultimo dia de cada trozo de la serie
    
#     """
#     def __init__(self, input_features=10, hidden_size=20, output_size=1, num_layers=1):
#         super(LSTMModel_parallel, self).__init__()
#         # LSTM layer
#         self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
#         # Linear layer en paralelo al LSTM
#         self.linear = nn.Linear(in_features=input_features, out_features=hidden_size)
#         # Capa lineal que recibe la concatenación de las salidas de LSTM y la capa lineal
#         self.output_linear = nn.Linear(in_features=(2*hidden_size)+input_features, out_features=output_size)

#     def forward(self, x):
#         # x debe tener la forma (batch_size, seq_lenght, input_features)
#         # Salida del LSTM
#         lstm_out, _ = self.lstm(x)  # lstm_out tiene la forma (batch_size, seq_lenght, hidden_size)
#         # Tomamos solo la última salida del LSTM para cada secuencia
#         lstm_out = lstm_out[:, -1, :]  # Ahora lstm_out tiene la forma (batch_size, hidden_size)

#         # Salida de la capa lineal
#         #opcion 1: solo el ultimo dia de la sequencia
#         mlp_output = self.linear(x[:, -1, :])  # Usamos solo el último elemento de la secuencia
        
#         # Concatenamos las salidas del LSTM y la capa lineal
#         combined_output = torch.cat((lstm_out, mlp_output, x[:, -1, :]), dim=1)

#         # Pasamos la salida combinada a través de la capa lineal final
#         final_output = self.output_linear(combined_output)

#         return final_output
class LSTMModel_parallel(nn.Module):
    def __init__(self, input_features=10, hidden_size=20, output_size=1, num_layers=1, dropout_rate=0.5):
        super(LSTMModel_parallel, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True, dropout=dropout_rate)
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        # Linear layer in parallel to LSTM
        self.linear = nn.Linear(in_features=input_features, out_features=hidden_size)
        # Batch normalization layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        # Activation function
        self.activation = nn.ReLU()  # You can use nn.Tanh() or nn.Sigmoid() as well
        # Output linear layer that receives the concatenation of LSTM and linear layer outputs
        self.output_linear = nn.Linear(in_features=(3*hidden_size)+input_features, out_features=output_size)

    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)
        # Take only the last output of the LSTM for each sequence
        lstm_out = lstm_out[:, -1, :]
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        # Linear layer output
        mlp_output = self.linear(x[:, -1, :])
        # Apply batch normalization
        mlp_output = self.batch_norm(mlp_output)
        # Apply activation function
        mlp_output = self.activation(mlp_output)
        # Concatenate LSTM and linear layer outputs
        combined_output = torch.cat((lstm_out, mlp_output, x[:, -1, :]), dim=1)
        # Final output
        final_output = self.output_linear(combined_output)
        return final_output
    
# Set hyperparameters
input_size = train_shape[-1]
print(f"Input size {input_size}")
hidden_size = 40
output_size = 1
num_layers = 2
learning_rate = 3e-4
num_epochs = 500

# Instantiate the models
lstm_model = LSTMModel_parallel(input_size, hidden_size, output_size, num_layers).to(device)
print(lstm_model)
# Define the loss function and optimizer
criterion_rmse = nn.MSELoss()  # RMSE
criterion_dir_acc = DirectionalAccuracyLoss()  

# Experimentar con diferentes optimizadores
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=learning_rate)
# optimizer_lstm = optim.AdamW(lstm_model.parameters(), lr=learning_rate)  # AdamW optimizer
# optimizer_lstm = optim.RMSprop(lstm_model.parameters(), lr=learning_rate)  # RMSprop optimizer

# Function to evaluate the model on validation data
def evaluate_model(model, val_seq, alpha=0.5):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for seq, labels in val_seq:
            seq, labels = seq.to(device), labels.to(device)
            y_pred = model(seq)
            loss_rmse = criterion_rmse(y_pred.reshape(-1,1), labels.reshape(-1,1))
            loss_dir_acc = criterion_dir_acc(y_pred.reshape(-1,1), labels.reshape(-1,1))
            loss = alpha * loss_rmse + (1 - alpha) * loss_dir_acc
            val_loss += loss.item()
        return val_loss / len(val_seq)

# Bucle de entrenamiento para el modelo LSTM
print("Entrenando el modelo LSTM en paralelo...")
train_losses_lstm = []
val_losses_lstm = []
for epoch in range(num_epochs):
    lstm_model.train()
    model_loss = 0.0
    alpha=0.8
    # entrenamiento
    for data_point in iter(entrenamiento_dataloader):
        # separar label y datos
        datos, label  = data_point       
        optimizer_lstm.zero_grad()
        y_pred = lstm_model(datos)
        loss_rmse = criterion_rmse(y_pred.reshape(-1, 1), label.reshape(-1, 1))
        loss_dir_acc = criterion_dir_acc(y_pred.reshape(-1,1), label.reshape(-1,1))
        loss = alpha * loss_rmse + (1 - alpha) * loss_dir_acc
        loss.backward()
        optimizer_lstm.step()
        model_loss += loss.item()

    train_loss = model_loss / len(entrenamiento_dataloader)
    train_losses_lstm.append(train_loss)
    val_loss = evaluate_model(lstm_model, validation_dataloader, alpha=alpha)
    val_losses_lstm.append(val_loss)

    if epoch % 3 == 0:
        print(f'Epoch {epoch} Train Loss {train_loss} Val Loss {val_loss}')

def predict(model, val_seq):
    model.eval()
    target = []
    predictions = []
    with torch.no_grad():
        for data_point in iter(val_seq):
            # separate label and data
            datos, label = data_point
            y_pred = model(datos)
            predictions.append(y_pred)
            target.append(label)

    # Convert lists to tensors
    predictions_tensor = torch.cat(predictions, dim=0)
    target_tensor = torch.cat(target, dim=0)

    # Calculate the accumulated error
    error = target_tensor - predictions_tensor
    accumulated_error = torch.cumsum(error, dim=0)

    return predictions_tensor, target_tensor, accumulated_error

###### Evaluaciones #####
def rolling_mape(predicciones, reales, ventana=20):
    """
    Calcula el Mean Absolute Percentage Error (MAPE) rodante.
    
    Args:
        predicciones (np.array): Array de predicciones.
        reales (np.array): Array de valores reales.
        ventana (int): Tamaño de la ventana para el cálculo rodante.
    
    Returns:
        np.array: Array con los valores de MAPE rodantes.
    """
    # Inicializa una lista vacía para almacenar los valores de MAPE rodantes
    rolling_mape = []
    # Itera sobre el rango de predicciones, aplicando una ventana
    for i in range(ventana, len(predicciones)):
        # Calcula el MAPE para la ventana actual y lo agrega a la lista
        mape = (np.abs(reales[i-ventana:i] - predicciones[i-ventana:i]) / np.abs(reales[i-ventana:i])).mean() * 100
        rolling_mape.append(mape)
    return rolling_mape

def exactitud_direccional(predicciones, reales):
    """
    Calcula la exactitud direccional, es decir, el porcentaje de veces que
    la predicción y el valor real tienen el mismo signo.
    
    Args:
        predicciones (np.array): Array de predicciones.
        reales (np.array): Array de valores reales.
    
    Returns:
        float: Porcentaje de exactitud direccional.
    """
    # Compara los signos de las diferencias consecutivas de predicciones y reales
    direccion_correcta = np.sign(predicciones[1:] - predicciones[:-1]) == np.sign(reales[1:] - reales[:-1])
    # Calcula y devuelve el porcentaje de veces que los signos coinciden
    return direccion_correcta.mean() * 100

def simular_pnl(predicciones, reales, capital_inicial=1000):
    """
    Simula el Profit and Loss (PnL) basado en las predicciones y los valores reales.
    
    Args:
        predicciones (np.array): Array de predicciones.
        reales (np.array): Array de valores reales.
        capital_inicial (float): Capital inicial para la simulación.
    
    Returns:
        np.array: Array con los valores de PnL simulados.
    """
    # Inicializa una lista con el capital inicial
    capital = [capital_inicial]
    # Itera sobre las predicciones para simular las transacciones
    for i in range(1, len(predicciones)):
        if predicciones[i] > predicciones[i-1]:  # Predecir un aumento
            capital.append(capital[-1] + (reales[i] / reales[i-1] - 1) * capital[-1])
        elif predicciones[i] < predicciones[i-1]:  # Predecir una disminución
            capital.append(capital[-1] - (reales[i] / reales[i-1] - 1) * capital[-1])
        else:
            capital.append(capital[-1])
    return np.array(capital)

def maximo_drawdown(pnl):
    """
    Calcula el máximo drawdown, que es la máxima pérdida desde un pico hasta un valle.
    
    Args:
        pnl (np.array): Array de Profit and Loss (PnL).
    
    Returns:
        float: Valor del máximo drawdown.
    """
    # Inicializa el pico y el máximo drawdown
    pico = pnl[0]
    max_drawdown = 0
    # Itera sobre los valores de PnL para encontrar el máximo drawdown
    for i in range(1, len(pnl)):
        if pnl[i] > pico:
            pico = pnl[i]
        drawdown = pico - pnl[i]
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


################

dataset.cambiar_modo('full')
full_dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=False)

predictions_lstm, targets_lstm, accumulated_error = predict(lstm_model, full_dataloader)

# Convert tensors to numpy arrays for plotting
predictions_lstm = predictions_lstm.reshape(-1).cpu().numpy()
targets_lstm = targets_lstm.reshape(-1).cpu().numpy()
accumulated_error = accumulated_error.reshape(-1).cpu().numpy()

# Calcular métricas adicionales
valores_rolling_mape = rolling_mape(predictions_lstm, targets_lstm)
exactitud_dir = exactitud_direccional(predictions_lstm, targets_lstm)
pnl = simular_pnl(predictions_lstm, targets_lstm)
valor_max_drawdown = maximo_drawdown(pnl)

# Imprimir métricas basadas en texto
print(f"Exactitud Direccional: {exactitud_dir:.2f}%")
print(f"Máximo Drawdown: {valor_max_drawdown:.2f}")

# Configurando una cuadrícula de subtramas 4x2
fig, axs = plt.subplots(4, 2, figsize=(18, 24))

# Graficando objetivos vs predicciones para LSTM
axs[0, 0].plot(targets_lstm, label='Objetivos Reales (LSTM)')
axs[0, 0].plot(predictions_lstm, label='Predicciones LSTM')
axs[0, 0].set_title('Objetivos Reales vs Predicciones LSTM')
axs[0, 0].set_xlabel('Paso de Tiempo')
axs[0, 0].set_ylabel('Valor Normalizado')
axs[0, 0].legend()

# Graficando la pérdida de entrenamiento vs validación para LSTM
axs[0, 1].plot(train_losses_lstm, label='Pérdida de Entrenamiento LSTM')
axs[0, 1].plot(val_losses_lstm, label='Pérdida de Validación LSTM')
axs[0, 1].set_title('Pérdida de Entrenamiento vs Validación LSTM')
axs[0, 1].set_xlabel('Época')
axs[0, 1].set_ylabel('Pérdida')
axs[0, 1].legend()

# Calcular el error acumulado medio
error_acumulado_medio = accumulated_error / np.arange(1, len(accumulated_error) + 1)

# Calcular el rango intercuartílico (IQR)
Q1 = np.percentile(error_acumulado_medio, 25)
Q3 = np.percentile(error_acumulado_medio, 75)
IQR = Q3 - Q1

# Definir límites para identificar outliers
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtrar outliers
error_acumulado_medio_sin_outliers = np.where(
    (error_acumulado_medio > limite_inferior) & (error_acumulado_medio < limite_superior),
    error_acumulado_medio,
    np.nan  # Puedes reemplazar los outliers con NaN si quieres conservar la forma del array
)

# Graficar el error acumulado medio sin outliers
axs[1, 0].plot(error_acumulado_medio_sin_outliers, label='Error Acumulado Medio (LSTM)')
axs[1, 0].set_title('Error Acumulado Medio con el Tiempo')
axs[1, 0].set_xlabel('Paso de Tiempo')
axs[1, 0].set_ylabel('Error Acumulado Medio')
axs[1, 0].legend()

# Graficando los retornos acumulativos
retornos_acumulativos_reales = (1 + targets_lstm).cumprod() - 1
retornos_acumulativos_predichos = (1 + predictions_lstm).cumprod() - 1
axs[1, 1].plot(retornos_acumulativos_reales, label='Retornos Acumulativos Reales')
axs[1, 1].plot(retornos_acumulativos_predichos, label='Retornos Acumulativos Predichos')
axs[1, 1].set_title('Retornos Acumulativos: Predichos vs. Reales')
axs[1, 1].set_xlabel('Paso de Tiempo')
axs[1, 1].set_ylabel('Retornos Acumulativos')
axs[1, 1].legend()

# Graficando el MAPE rodante
axs[2, 0].plot(valores_rolling_mape, label='MAPE Rodante')
axs[2, 0].set_title('Error Porcentual Absoluto Medio Rodante')
axs[2, 0].set_xlabel('Paso de Tiempo')
axs[2, 0].set_ylabel('MAPE')
axs[2, 0].legend()

# Graficando la distribución del error
errores = targets_lstm - predictions_lstm
axs[2, 1].hist(errores, bins=50, alpha=0.7, color='r')
axs[2, 1].set_xlabel('Error de Predicción')
axs[2, 1].set_ylabel('Frecuencia')
axs[2, 1].set_title('Distribución de Errores de Predicción')

# Graficando el PnL simulado
axs[3, 0].plot(pnl, label='PnL Simulado')
axs[3, 0].set_title('Ganancias y Pérdidas Simuladas')
axs[3, 0].set_xlabel('Paso de Tiempo')
axs[3, 0].set_ylabel('PnL')
axs[3, 0].legend()

# El último subtrama (3, 1) se deja vacío
axs[3, 1].axis('off')

# Ajustar el diseño para evitar la superposición
plt.tight_layout()
plt.show()
