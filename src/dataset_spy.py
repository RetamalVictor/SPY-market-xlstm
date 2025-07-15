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
        If normalization is enabled (normalize=True), all columns (except 'spy_sign') are normalized.
        For multi-task mode (multi_task=True), the target is a tensor [spy_sign, spy] (classification and regression).
    Args:
        config (DatasetConfig): Configuration object containing dataset parameters. Must include:
            - file_path: path to the Excel file.
            - seq_len: sequence length.
            - mode: "train", "val", or "test".
            - device: device to load tensors on.
            - train_ratio: ratio for training split.
            - val_ratio: ratio for validation split.
            - norm_param_path: file path to save (or load) normalization parameters.
            - classification (bool): If True, use spy_sign as target.
            - multi_task (bool): If True, return both classification and regression targets.
            - normalize (bool): If True, apply normalization to the data. Default is True.
    Raises:
        ValueError: If mode is not one of 'train', 'val', or 'test' or if the 'spy' column is missing.
    Return:
        An instance of TradingDataset that returns a tuple (input_sequence, target) for each sample,
        where input_sequence is a torch.Tensor of shape (seq_len, num_features) and target is either:
          - A scalar (if single-task) or
          - A torch.Tensor of shape [2] (if multi_task=True).
    """
    def __init__(self, config: "DatasetConfig"):
        super(TradingDataset, self).__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.mode = config.mode.lower()
        self.norm_params_path = config.norm_param_path
        self.norm_params = None  # Will hold normalization parameters if enabled

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError("mode must be one of 'train', 'val', or 'test'")

        # Load data from three sheets
        features = pd.read_excel(config.file_path, sheet_name='features', skiprows=[1])
        sectores = pd.read_excel(config.file_path, sheet_name='sectores', skiprows=[1])
        paises = pd.read_excel(config.file_path, sheet_name='paises', skiprows=[1])

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

        # Convert all to float and create classification target before normalization
        self.data = self.data.astype(float)
        self.data['spy_sign'] = (self.data['spy'] > 0).astype(float)

        # Adjust if the number of features is odd by adding a dummy column
        if self.data.shape[1] % 2 != 0:
            self.data['dummy'] = 0.0

        # Apply normalization only if enabled (default True)
        if getattr(self.config, "normalize", True):
            columns_to_normalize = [col for col in self.data.columns if col not in ['spy_sign']]
            if self.mode == 'train':
                norm_params = {}
                for col in columns_to_normalize:
                    col_min = self.data[col].min()
                    col_max = self.data[col].max()
                    norm_params[col] = {"min": float(col_min), "max": float(col_max)}
                    self.data[col] = 2 * (self.data[col] - col_min) / (col_max - col_min) - 1
                # Save and store normalization parameters
                with open(self.norm_params_path, "w") as f:
                    json.dump(norm_params, f, indent=4)
                self.norm_params = norm_params
            else:
                if not os.path.exists(self.norm_params_path):
                    raise ValueError(f"Normalization parameters file {self.norm_params_path} not found for {self.mode} mode.")
                with open(self.norm_params_path, "r") as f:
                    norm_params = json.load(f)
                for col in columns_to_normalize:
                    if col in norm_params:
                        col_min = norm_params[col]["min"]
                        col_max = norm_params[col]["max"]
                        self.data[col] = 2 * (self.data[col] - col_min) / (col_max - col_min) - 1
                self.norm_params = norm_params

        # Fill NaNs
        self.data.fillna(0, inplace=True)
        self.data_array = self.data.to_numpy()

    def denormalize_spy(self, spy_norm: float) -> float:
        """
        Description:
            Reverts the normalization for a given spy value.
        Args:
            spy_norm (float): Normalized spy value.
        Returns:
            float: The denormalized spy value.
        """
        if self.norm_params is None or "spy" not in self.norm_params:
            raise ValueError("Normalization parameters for 'spy' are not available.")
        spy_min = self.norm_params["spy"]["min"]
        spy_max = self.norm_params["spy"]["max"]
        return ((spy_norm + 1) / 2) * (spy_max - spy_min) + spy_min

    def __len__(self):
        return len(self.data_array) - self.seq_len

    def __getitem__(self, idx: int):
        """
        Description:
            Retrieves the sample at the given index.
        Args:
            idx (int): Index of the sample.
        Raises:
            IndexError: If index is out of range.
        Returns:
            tuple: (input_sequence, target) where input_sequence is a tensor of shape (seq_len, num_features)
                   and target is either a scalar (if single-task) or a tensor of shape [2] (if multi_task=True).
        """
        if idx < 0 or idx >= self.__len__():
            raise IndexError("Index out of range")
        x = self.data_array[idx : idx + self.seq_len, :]
        spy_index = self.data.columns.get_loc('spy')
        spy_sign_index = self.data.columns.get_loc('spy_sign')
        sign_val = self.data_array[idx + self.seq_len, spy_sign_index]
        spy_val = self.data_array[idx + self.seq_len, spy_index]
        if getattr(self.config, "multi_task", False):
            target = torch.tensor([sign_val, abs(spy_val)], dtype=torch.float32)
        else:
            if self.config.classification:
                target = torch.tensor(sign_val, dtype=torch.float32)
            else:
                target = torch.tensor(spy_val, dtype=torch.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        return x_tensor, target

# Example usage remains unchanged.
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from config import DatasetConfig

    config = DatasetConfig(
        file_path="data/seriesDiariasNumbersVictor.xlsx",
        seq_len=5,
        mode="test",     # "train", "val", or "test"
        device="cuda",
        norm_param_path="data/test_norm.json",
        classification=True,
        multi_task=True,  # Enable multi-task mode
        normalize=True    # Set to False to disable normalization
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
