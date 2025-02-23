import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import os

class TradingDataset(Dataset):
    """
    Description:
        A PyTorch Dataset for sequential trading data. This dataset reads an Excel file containing three sheets,
        merges them on the date column ("Unnamed: 0"), splits the data into training, validation, and test sets,
        normalizes each column, and generates samples where the input is a sequence of trading data from t-n to t-1
        (for all features) and the target is the 'spy' value at time t. In train mode, the normalization parameters
        are computed and saved to the provided norm_param_path; in val/test mode, the parameters are loaded from that file.
    Args:
        config (DatasetConfig): Configuration object containing dataset parameters. Must include:
            - file_path: path to the Excel file.
            - seq_len: sequence length.
            - mode: "train", "val", or "test".
            - device: device to load tensors on (this is used only for non-dataset operations).
            - train_ratio: ratio for training split.
            - val_ratio: ratio for validation split.
            - norm_param_path: file path to save (or load) normalization parameters.
    Raises:
        ValueError: If mode is not one of 'train', 'val', or 'test' or if the 'spy' column is missing.
    Return:
        An instance of TradingDataset that returns a tuple (input_sequence, target) for each sample,
        where input_sequence is a torch.Tensor of shape (seq_len, num_features) on the CPU and
        target is a torch.Tensor scalar representing the 'spy' value at time t on the CPU.
    """
    def __init__(self, config: 'DatasetConfig'):
        super(TradingDataset, self).__init__()
        self.seq_len = config.seq_len
        self.mode = config.mode.lower()
        # Note: We now return CPU tensors, so we don't move data to config.device here.
        self.norm_params_path = config.norm_param_path  # Path to normalization parameters JSON file
        if self.mode not in ['train', 'val', 'test']:
            raise ValueError("mode must be one of 'train', 'val', or 'test'")
        features = pd.read_excel(config.file_path, sheet_name='features', skiprows=[1])
        sectores = pd.read_excel(config.file_path, sheet_name='sectores', skiprows=[1])
        paises = pd.read_excel(config.file_path, sheet_name='paises', skiprows=[1])
        features['Unnamed: 0'] = pd.to_datetime(features['Unnamed: 0'])
        sectores['Unnamed: 0'] = pd.to_datetime(sectores['Unnamed: 0'])
        paises['Unnamed: 0'] = pd.to_datetime(paises['Unnamed: 0'])
        df = features.merge(sectores, on='Unnamed: 0', how='inner').merge(paises, on='Unnamed: 0', how='inner')
        df.sort_values('Unnamed: 0', inplace=True)
        df.reset_index(drop=True, inplace=True)
        total_len = len(df)
        train_end = int(total_len * config.train_ratio)
        val_end = train_end + int(total_len * config.val_ratio)
        if self.mode == 'train':
            self.data = df.iloc[:train_end].reset_index(drop=True)
        elif self.mode == 'val':
            self.data = df.iloc[train_end:val_end].reset_index(drop=True)
        else:  # test
            self.data = df.iloc[val_end:].reset_index(drop=True)
        self.data = self.data.drop(columns=['Unnamed: 0']).reset_index(drop=True)
        if 'spy' not in self.data.columns:
            raise ValueError("The dataset must contain a 'spy' column as target")
        self.data = self.data.astype(float)

        # Adjust if the number of features is odd by adding a dummy column filled with zeros.
        if self.data.shape[1] % 2 != 0:
            self.data['dummy'] = 0.0

        # Normalization
        if self.mode == 'train':
            norm_params = {}
            for col in self.data.columns:
                col_min = self.data[col].min()
                col_max = self.data[col].max()
                norm_params[col] = {"min": float(col_min), "max": float(col_max)}
                # Normalize: norm(x) = 2*(x - min)/(max - min) - 1
                self.data[col] = 2 * (self.data[col] - col_min) / (col_max - col_min) - 1
            # Save normalization parameters to the specified JSON file
            with open(self.norm_params_path, "w") as f:
                json.dump(norm_params, f, indent=4)
        else:
            if not os.path.exists(self.norm_params_path):
                raise ValueError(f"Normalization parameters file {self.norm_params_path} not found for {self.mode} mode.")
            with open(self.norm_params_path, "r") as f:
                norm_params = json.load(f)
            for col in self.data.columns:
                if col in norm_params:
                    col_min = norm_params[col]["min"]
                    col_max = norm_params[col]["max"]
                    self.data[col] = 2 * (self.data[col] - col_min) / (col_max - col_min) - 1
        self.data.fillna(0, inplace=True)
        self.data_array = self.data.to_numpy()

    def __len__(self):
        return len(self.data_array) - self.seq_len

    def __getitem__(self, idx: int):
        """
        Description:
            Retrieves the sample at the given index.
        Args:
            idx (int): Index of the sample.
        Raises:
            IndexError: If idx is out of range.
        Return:
            tuple: A tuple (input_sequence, target) where input_sequence is a torch.Tensor of shape (seq_len, num_features)
                   containing trading data from t-n to t-1 on the CPU, and target is a torch.Tensor scalar
                   representing the 'spy' value at time t on the CPU.
        """
        if idx < 0 or idx >= self.__len__():
            raise IndexError("Index out of range")
        x = self.data_array[idx: idx + self.seq_len, :]
        spy_index = self.data.columns.get_loc('spy')
        target = self.data_array[idx + self.seq_len, spy_index]
        # Return CPU tensors; data will be moved to GPU later if needed.
        x_tensor = torch.tensor(x, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return x_tensor, target_tensor

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from config import DatasetConfig
    config = DatasetConfig(
        file_path=r"data\other\seriesDiariasNumbersVictor.xlsx",
        seq_len=5,       # Change the window size as needed.
        mode="test",    # Can be "train", "val", or "test"
        device="cuda",     # Change to "cuda" if you have a GPU available.
        norm_param_path="data/other/test_norm.json"
    )

    # Create the TradingDataset instance.
    dataset = TradingDataset(config)

    pin_memory = True if config.device == "cpu" else False
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, pin_memory=pin_memory)

    # Iterate over the DataLoader and print some information about each batch.
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print("  Inputs shape:", inputs.shape)
        print("  Targets shape:", targets.shape)
        print(f"Target are  {targets}")
        # Process only a few batches for demonstration.
        if batch_idx >= 3:
            break