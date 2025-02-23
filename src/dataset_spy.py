import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import os

class TradingDataset(Dataset):
    """
    Description:
        A PyTorch Dataset for sequential trading data. This dataset reads an Excel file containing three sheets,
        merges them on the date column ("Unnamed: 0"), splits the data into training, validation, and test sets.
        It creates a new column 'spy_sign' (0 or 1) before normalization so that classification targets remain intact.
        The 'spy' column is then normalized only if needed. If classification=True, 'spy_sign' is used as the target;
        otherwise, the numeric 'spy' value is used.
    Args:
        config (DatasetConfig): Configuration object containing dataset parameters. Must include:
            - file_path: path to the Excel file.
            - seq_len: sequence length.
            - mode: "train", "val", or "test".
            - device: device to load tensors on (this is used only for non-dataset operations).
            - train_ratio: ratio for training split.
            - val_ratio: ratio for validation split.
            - norm_param_path: file path to save (or load) normalization parameters.
            - classification (bool): If True, use spy_sign as target; otherwise use numeric spy.
    Raises:
        ValueError: If mode is not one of 'train', 'val', or 'test' or if the 'spy' column is missing.
    Return:
        An instance of TradingDataset that returns a tuple (input_sequence, target) for each sample,
        where input_sequence is a torch.Tensor of shape (seq_len, num_features) on the CPU and
        target is a torch.Tensor (scalar) representing the sign or value of 'spy' at time t.
    """
    def __init__(self, config: 'DatasetConfig'):
        super(TradingDataset, self).__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.mode = config.mode.lower()
        self.norm_params_path = config.norm_param_path

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError("mode must be one of 'train', 'val', or 'test'")

        # Load data from three sheets
        features = pd.read_excel(config.file_path, sheet_name='features', skiprows=[1])
        sectores = pd.read_excel(config.file_path, sheet_name='sectores', skiprows=[1])
        paises = pd.read_excel(config.file_path, sheet_name='paises', skiprows=[1])

        # Convert date column
        features['Unnamed: 0'] = pd.to_datetime(features['Unnamed: 0'])
        sectores['Unnamed: 0'] = pd.to_datetime(sectores['Unnamed: 0'])
        paises['Unnamed: 0'] = pd.to_datetime(paises['Unnamed: 0'])

        # Merge all into one DataFrame
        df = (
            features
            .merge(sectores, on='Unnamed: 0', how='inner')
            .merge(paises, on='Unnamed: 0', how='inner')
        )
        df.sort_values('Unnamed: 0', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Split into train/val/test
        total_len = len(df)
        train_end = int(total_len * config.train_ratio)
        val_end = train_end + int(total_len * config.val_ratio)
        if self.mode == 'train':
            self.data = df.iloc[:train_end].reset_index(drop=True)
        elif self.mode == 'val':
            self.data = df.iloc[train_end:val_end].reset_index(drop=True)
        else:  # test
            self.data = df.iloc[val_end:].reset_index(drop=True)

        # Drop date column
        self.data = self.data.drop(columns=['Unnamed: 0']).reset_index(drop=True)

        if 'spy' not in self.data.columns:
            raise ValueError("The dataset must contain a 'spy' column as target")

        # Convert all to float
        self.data = self.data.astype(float)

        # Create a sign column *before* normalization
        self.data['spy_sign'] = (self.data['spy'] > 0).astype(float)

        # Adjust if the number of features is odd by adding a dummy column
        if self.data.shape[1] % 2 != 0:
            self.data['dummy'] = 0.0

        # Normalize all columns except 'spy_sign' (already 0 or 1)
        columns_to_normalize = [
            col for col in self.data.columns
            if col not in ['spy_sign']  # Skip sign column
        ]

        if self.mode == 'train':
            norm_params = {}
            for col in columns_to_normalize:
                col_min = self.data[col].min()
                col_max = self.data[col].max()
                norm_params[col] = {"min": float(col_min), "max": float(col_max)}
                # Normalize: norm(x) = 2*(x - min)/(max - min) - 1
                self.data[col] = 2 * (self.data[col] - col_min) / (col_max - col_min) - 1

            # Save normalization parameters
            with open(self.norm_params_path, "w") as f:
                json.dump(norm_params, f, indent=4)

        else:
            # Load normalization parameters
            if not os.path.exists(self.norm_params_path):
                raise ValueError(f"Normalization parameters file {self.norm_params_path} not found for {self.mode} mode.")
            with open(self.norm_params_path, "r") as f:
                norm_params = json.load(f)
            for col in columns_to_normalize:
                if col in norm_params:
                    col_min = norm_params[col]["min"]
                    col_max = norm_params[col]["max"]
                    self.data[col] = 2 * (self.data[col] - col_min) / (col_max - col_min) - 1

        # Fill NaNs
        self.data.fillna(0, inplace=True)

        # Convert final DataFrame to numpy for indexing
        self.data_array = self.data.to_numpy()

    def __len__(self):
        return len(self.data_array) - self.seq_len

    def __getitem__(self, idx: int):
        """
        Description:
            Retrieves the sample at the given index.
        Args:
            idx (int): Index of the sample.
        Return:
            tuple: (input_sequence, target) where input_sequence is shape (seq_len, num_features),
                   and target is scalar (spy_sign if classification=True, else numeric spy).
        """
        if idx < 0 or idx >= self.__len__():
            raise IndexError("Index out of range")

        # Slice the input window
        x = self.data_array[idx : idx + self.seq_len, :]

        # Get the target from time t (the next index after the window)
        spy_index = self.data.columns.get_loc('spy')
        spy_sign_index = self.data.columns.get_loc('spy_sign')

        if self.config.classification:
            # Use spy_sign column as the target
            target = self.data_array[idx + self.seq_len, spy_sign_index]
        else:
            # Use numeric spy value
            target = self.data_array[idx + self.seq_len, spy_index]

        x_tensor = torch.tensor(x, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return x_tensor, target_tensor


# Example usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from config import DatasetConfig

    config = DatasetConfig(
        file_path="data/other/seriesDiariasNumbersVictor.xlsx",
        seq_len=5,
        mode="test",     # "train", "val", or "test"
        device="cuda",   # or "cpu"
        norm_param_path="data/other/test_norm.json",
        classification=True  # Switch between True/False
    )

    dataset = TradingDataset(config)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print("  Inputs shape:", inputs.shape)
        print("  Targets shape:", targets.shape)
        print("  Targets:", targets)
        if batch_idx >= 3:
            break
