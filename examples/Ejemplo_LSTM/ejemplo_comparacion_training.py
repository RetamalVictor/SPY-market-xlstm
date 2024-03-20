import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Check if a GPU is available and set the device accordingly
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Load the dataset
df = pd.read_csv(r'examples\lstmcell\time_series_data.csv')

# Select features and target
features = df.drop(columns=['x', 'combined_series_correlated_disturbed'])
target = df['combined_series_correlated_disturbed']

# Normalize the data
scaler_features = MinMaxScaler(feature_range=(-1, 1))
scaler_target = MinMaxScaler(feature_range=(-1, 1))
features_normalized = scaler_features.fit_transform(features)
target_normalized = scaler_target.fit_transform(target.values.reshape(-1, 1))

# Convert to PyTorch tensors
features_normalized = torch.FloatTensor(features_normalized).to(device)
target_normalized = torch.FloatTensor(target_normalized).view(-1).to(device)

# Define a function to create input-output sequence pairs
def create_inout_sequences(input_features, input_target, tw):
    inout_seq = []
    L = len(input_target)
    for i in range(L - tw):
        train_seq = input_features[i:i + tw]
        train_label = input_target[i + tw:i + tw + 1]
        inout_seq.append([train_seq, train_label])
    return inout_seq

# Create input-output sequence pairs
train_window = 12
train_inout_seq = create_inout_sequences(features_normalized[:-200], target_normalized[:-200], train_window)
val_inout_seq = create_inout_sequences(features_normalized[-200:], target_normalized[-200:], train_window)

# [longitud_sequ, muestra, numer_feat]
# [12, N, 8]

# [12, batch_size, 8]

# Define the LSTMCell-based model
# Esta clase la he creado yo. No es de torch
class LSTMCellModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMCellModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell1 = nn.LSTMCell(input_size, hidden_size).to(device)
        self.lstm_cell2 = nn.LSTMCell(hidden_size, hidden_size).to(device)
        self.linear = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, input_seq):
        outputs = []
        hx1 = torch.zeros( self.hidden_size).to(device)
        cx1 = torch.zeros( self.hidden_size).to(device)
        hx2 = torch.zeros( self.hidden_size).to(device)
        cx2 = torch.zeros( self.hidden_size).to(device)
        for i in range(input_seq.size(0)):
            hx1, cx1 = self.lstm_cell1(input_seq[i], (hx1, cx1))
            hx2, cx2 = self.lstm_cell2(hx1, (hx2, cx2))
            outputs.append(hx2)
        outputs = torch.stack(outputs)
        output = self.linear(hx2)
        return output  # Return only the last prediction

# Define the LSTM-based model
# Este tambien lo he nombrado yo
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers).to(device)
        self.linear = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, input_seq):
        h0 = torch.zeros(self.num_layers, self.hidden_size, dtype=torch.float32).to(device)
        c0 = torch.zeros(self.num_layers, self.hidden_size, dtype=torch.float32).to(device)

        lstm_out, (hn, cn) = self.lstm(input_seq, (h0, c0))
        output = self.linear(lstm_out[-1])
        return output

# Set hyperparameters
input_size = features.shape[1]
hidden_size = 40
output_size = 1
num_layers = 2
learning_rate = 3e-4
num_epochs = 15

# Instantiate the models
lstm_cell_model = LSTMCellModel(input_size, hidden_size, output_size).to(device)
lstm_model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer_cell = optim.Adam(lstm_cell_model.parameters(), lr=learning_rate)
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

# Training loop for the LSTMCell-based model
print("Training LSTMCell-based model...")
train_losses_cell = []
val_losses_cell = []
for epoch in range(num_epochs):
    lstm_cell_model.train()
    model_loss = 0.0
    for seq, labels in train_inout_seq:
        seq, labels = seq.to(device), labels.to(device)
        optimizer_cell.zero_grad()
        y_pred = lstm_cell_model(seq)
        loss = criterion(y_pred.reshape(-1,1), labels.reshape(-1,1))
        loss.backward()
        optimizer_cell.step()
        model_loss += loss.item()

    train_loss = model_loss / len(train_inout_seq)
    train_losses_cell.append(train_loss)
    val_loss = evaluate_model(lstm_cell_model, val_inout_seq)
    val_losses_cell.append(val_loss)

    if epoch % 3 == 0:
        print(f'Epoch {epoch} Train Loss {train_loss} Val Loss {val_loss}')

# Training loop for the LSTM-based model
print("Training LSTM-based model...")
train_losses_lstm = []
val_losses_lstm = []
for epoch in range(num_epochs):
    lstm_model.train()
    model_loss = 0.0
    for seq, labels in train_inout_seq:
        seq, labels = seq.to(device), labels.to(device)  # Reshape labels to match output shape
        optimizer_lstm.zero_grad()
        y_pred = lstm_model(seq)
        loss = criterion(y_pred.reshape(-1,1), labels.reshape(-1,1))
        loss.backward()
        optimizer_lstm.step()
        model_loss += loss.item()

    train_loss = model_loss / len(train_inout_seq)
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


inout_seq = create_inout_sequences(features_normalized, target_normalized, train_window)

predictions_cell ,targets_cell=predict(lstm_cell_model, inout_seq)
predictions_lstm ,targets_lstm=predict(lstm_model, inout_seq)

# Setting up a 2x2 subplot grid
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

# Plotting targets vs predictions for LSTMCell
axs[0, 0].plot(targets_cell, label='Ground Truth (LSTMCell)')
axs[0, 0].plot(predictions_cell, label='LSTMCell Predictions')
axs[0, 0].set_title('Ground Truth vs LSTMCell Predictions')
axs[0, 0].set_xlabel('Time Step')
axs[0, 0].set_ylabel('Normalized Value')
axs[0, 0].legend()

# Plotting targets vs predictions for LSTM
axs[0, 1].plot(targets_lstm, label='Ground Truth (LSTM)')
axs[0, 1].plot(predictions_lstm, label='LSTM Predictions')
axs[0, 1].set_title('Ground Truth vs LSTM Predictions')
axs[0, 1].set_xlabel('Time Step')
axs[0, 1].set_ylabel('Normalized Value')
axs[0, 1].legend()

# Plotting training vs validation loss for LSTMCell
axs[1, 0].plot(train_losses_cell, label='LSTMCell Train Loss')
axs[1, 0].plot(val_losses_cell, label='LSTMCell Val Loss')
axs[1, 0].set_title('LSTMCell Training vs Validation Loss')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].legend()

# Plotting training vs validation loss for LSTM
axs[1, 1].plot(train_losses_lstm, label='LSTM Train Loss')
axs[1, 1].plot(val_losses_lstm, label='LSTM Val Loss')
axs[1, 1].set_title('LSTM Training vs Validation Loss')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Loss')
axs[1, 1].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()