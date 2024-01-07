import torch
from torch.utils.data import Dataset, DataLoader

def reshape_data_for_lstm(data, sequence_length, pad_value=0.0, padding=False):
    """
    Reshape time series data for LSTM input, with optional padding for shorter sequences.

    Parameters:
    data (torch.Tensor): The time series data.
    sequence_length (int): The number of time steps in each input sequence.
    pad_value (float): The value to use for padding shorter sequences.
    padding (bool): Whether to pad sequences to ensure uniform length.

    Returns:
    X, y (torch.Tensor, torch.Tensor): Tensors for LSTM input and target output.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length + 1):
        seq = data[i:i + sequence_length]  # Sequence of features

        # Check if sequence is shorter than sequence_length and apply padding if needed
        if padding and len(seq) < sequence_length:
            padding_length = sequence_length - len(seq)
            pad = torch.full((padding_length, data.shape[1]), pad_value)
            seq = torch.cat([pad, seq], dim=0)

        target = data[i + sequence_length] if i + sequence_length < len(data) else torch.full((data.shape[1],), pad_value)
        X.append(seq)
        y.append(target)

    X = torch.stack(X)
    y = torch.stack(y)
    return X, y

# Example usage
data = torch.rand(12, 3)  # Replace with your actual data
sequence_length = 5
X, y = reshape_data_for_lstm(data, sequence_length, padding=False)
print(X.shape, y.shape)  # torch.Size([282, 10, 12]) torch.Size([282, 12]
print(X)
X = X.transpose(0,1)
print(X.shape, y.shape)  # torch.Size([282, 10, 12]) torch.Size([282, 12]
print(X)
# Rest of the code for TimeSeriesDataset and DataLoader remains the same
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, train_ratio=0.75, validation_ratio=0.15):
        # Initialize an empty dataset
        self.X = torch.Tensor()
        self.Y = torch.Tensor()
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio

    def create_time_series_dataset(self, data, features, target, look_back=1):
        dataX, dataY = [], []
        for i in range(len(data) - look_back):
            a = data[features].iloc[i:(i + look_back)].values
            dataX.append(a)
            dataY.append(data[target].iloc[i + look_back])
        return np.array(dataX), np.array(dataY)

    def add_data(self, data, features, target, look_back=1):
        # Create the time series dataset from numpy arrays
        X_time_series, y_time_series = self.create_time_series_dataset(data, features, target, look_back)

        # Convert the numpy arrays to PyTorch tensors and shuffle them
        X_tensor, Y_tensor = self.prepare_data_for_pytorch((X_time_series, y_time_series))

        # Concatenate the new data with the existing tensors
        if self.X.nelement() == 0:
            self.X = X_tensor
            self.Y = Y_tensor
        else:
            self.X = torch.cat((self.X, X_tensor), dim=0)
            self.Y = torch.cat((self.Y, Y_tensor), dim=0)

        # Update the indices for train/validation split
        num_days = self.X.size(0)
        self.train_index = int(self.train_ratio * num_days)
        self.validation_index = int((self.train_ratio + self.validation_ratio) * num_days)

    def prepare_data_for_pytorch(self, dataset_numpy):
        # Convert the current dataset to PyTorch tensors
        X_tensor, Y_tensor = self.convert_to_torch(*dataset_numpy)

        # Shuffle the data
        indices = torch.randperm(X_tensor.size(0))
        X_tensor = X_tensor[indices]
        Y_tensor = Y_tensor[indices]

        return X_tensor, Y_tensor

    @staticmethod
    def convert_to_torch(X, Y):
        # Reshape X to have 3 dimensions if it does not have
        if len(X.shape) != 3:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        # Convert numpy arrays to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).transpose(1, 2)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
        return X_tensor, Y_tensor

    def switch_dataset(self, mode='train'):
        # Split the dataset based on the mode
        if mode == 'train':
            self.current_X = self.X[:self.train_index]
            self.current_Y = self.Y[:self.train_index]
        elif mode == 'validation':
            self.current_X = self.X[self.train_index:self.validation_index]
            self.current_Y = self.Y[self.train_index:self.validation_index]
        elif mode == 'test':
            self.current_X = self.X[self.validation_index:]
            self.current_Y = self.Y[self.validation_index:]
        else:
            raise ValueError("Invalid mode. Choose 'train', 'validation', or 'test'.")

    def __len__(self):
        return len(self.current_Y)

    def __getitem__(self, idx):
        return self.current_X[idx], self.current_Y[idx]

    def get_data(self):
      return self.current_X, self.current_Y

# Create an instance of the dataset
time_series_dataset = TimeSeriesDataset()
# Dummy DataFrame to test the class methods
df_dummy = pd.DataFrame({
    'var_x1': np.random.randn(100),
    'var_x2': np.random.randn(100),
    'var_y': np.random.randn(100)
})
# Add data to the dataset
features = ['var_x1', 'var_x2']
target = 'var_y'
time_series_dataset.add_data(df_dummy, features, target, look_back=5)

# Switch to train dataset
time_series_dataset.switch_dataset(mode='train')
X_tensor, Y_tensor = time_series_dataset.get_data()
# Output shapes for confirmation
print(X_tensor.shape, Y_tensor.shape)

print("test")
time_series_dataset.switch_dataset(mode="test")
X_tensor, Y_tensor = time_series_dataset.get_data()
# Output shapes for confirmation
print(X_tensor.shape, Y_tensor.shape)