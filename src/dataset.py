import pandas as pd
from typing import Tuple
import torch
from torch.utils.data import Dataset

from src.config import DatasetConfig

class CustomTradingSequenceDataset(Dataset):
    def __init__(self, data_file: str, config: 'DatasetConfig'):
        """
        Description: Handles creation of a time-series dataset for trading bar data.
        Args:
            data_file (str): Path to CSV or Parquet file containing trading data.
            config (DatasetConfig): Configuration object with dataset settings.
        Raises:
            ValueError: If prediction_horizon is invalid or not enough rows for configuration.
        Return: 
            CustomTradingSequenceDataset instance.
        """
        self.config = config
        self.transform = config.transform
        self.target_transform = config.target_transform

        #############################
        # 1. Load the Preprocessed File
        #############################
        print(f"Loading preprocessed dataset from {data_file}...")
        if data_file.endswith(".parquet"):
            df = pd.read_parquet(data_file)
        else:
            df = pd.read_csv(data_file, memory_map=True)

        initial_len = len(df)
        print(f"Read {initial_len} rows from file.")

        #############################
        # 2. Parse and Clean Datetime
        #############################
        df["DATETIME"] = pd.to_datetime(
            df["DTYYYYMMDD"] + df["TIME"],
            format="%Y%m%d%H%M%S",
            errors="coerce"
        )
        df.dropna(subset=["DATETIME"], inplace=True)
        df.sort_values(by="DATETIME", inplace=True)
        df.reset_index(drop=True, inplace=True)
        after_drop_len = len(df)
        print(f"Dropped {initial_len - after_drop_len} rows due to invalid datetime parsing.")

        #############################
        # 3. Convert Config Times
        #############################
        # Convert sequence_length_minutes to number of bars.
        self.sequence_length_bars = config.sequence_length_minutes // config.time_interval_minutes

        # For multi-target prediction we assume the preprocessed file already has target columns:
        # "target_1", "target_2", ..., "target_20". (In your preprocessing script, these were computed.)
        forecast_horizon = 18  # Fixed to 20 targets

        #############################
        # 4. Prepare Features, Targets & Datetimes
        #############################
        # Define which columns are features.
        feature_cols = ["inc", "OPEN", "HIGH", "LOW", "CLOSE", "TIME_DIFF"]
        # Define target columns (next 20 inc values).
        target_cols = [f"target_{i}" for i in range(1, forecast_horizon + 1)]
        
        self.features = df[feature_cols].to_numpy(dtype=float)
        self.targets = df[target_cols].to_numpy(dtype=float)
        self.datetimes = df["DATETIME"].values

        total_rows = len(self.features)
        print(f"Total rows after cleaning: {total_rows}")

        #############################
        # 5. Define Valid Indices
        #############################
        # We require that an input sequence of length sequence_length_bars exists,
        # and that the corresponding target row (from which the target vector is taken)
        # is available. We choose to take the target vector from the last row of the input window.
        #
        # That is, if an input window is from index i to i+sequence_length_bars-1,
        # then the target will be the vector stored in self.targets[i+sequence_length_bars-1].
        #
        # Valid indices: all i such that i + sequence_length_bars - 1 < total_rows.
        self.valid_indices = list(range(total_rows - self.sequence_length_bars + 1))
        self.total_samples = len(self.valid_indices)
        print(f"Initialized dataset with {self.total_samples} valid samples.")

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a valid sequence and its corresponding target vector.
        The target is taken from the row at the end of the input sequence,
        which contains the next 20 inc values.

        Args:
            idx (int): The index of the dataset sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (input_sequence, target_vector)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        start_idx = self.valid_indices[idx]
        end_of_input = start_idx + self.sequence_length_bars

        # Input sequence: shape [sequence_length_bars, num_features]
        input_seq = self.features[start_idx:end_of_input]
        input_seq = torch.tensor(input_seq, dtype=torch.float32)

        # Target: from the last row of the window, a vector of length 20.
        target = self.targets[end_of_input - 1]  # shape (20,)
        target = torch.tensor(target, dtype=torch.float32)

        if self.transform:
            input_seq = self.transform(input_seq)
        if self.target_transform:
            target = self.target_transform(target)
        return input_seq, target

    #############################
    # 6. Debug Methods
    #############################
    def debug_sequence_datetimes(self, idx: int):
        """
        Print the DATETIME and TIME_DIFF for each bar in the input sequence at dataset index `idx`.
        Args:
            idx (int): Index of the dataset sample to debug.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        start_idx = self.valid_indices[idx]
        end_of_input = start_idx + self.sequence_length_bars

        seq_datetimes = self.datetimes[start_idx: end_of_input]
        time_diffs = self.features[start_idx: end_of_input, 5]
        print(f"\n[DEBUG] Sequence datetimes & time differences for dataset index {idx}:")
        for i, (dt, td) in enumerate(zip(seq_datetimes, time_diffs)):
            print(f"  {i:2d} -> {dt} | Time diff: {td:.1f} min")

    def find_sequences_with_large_time_diff(self, threshold: float = 5.0):
        """
        Iterates through all possible sequences and prints those with any TIME_DIFF > threshold.
        Args:
            threshold (float): The time difference threshold in minutes.
        """
        print(f"\nScanning for sequences with any TIME_DIFF > {threshold} minutes...")
        total_rows = len(self.features)
        limit = total_rows - self.sequence_length_bars + 1
        for start_idx in range(limit):
            time_diffs = self.features[start_idx:start_idx + self.sequence_length_bars, -1]
            if (time_diffs > threshold).any():
                seq_datetimes = self.datetimes[start_idx:start_idx + self.sequence_length_bars]
                print(f"\nSequence starting at index {start_idx}:")
                for i, (dt, td) in enumerate(zip(seq_datetimes, time_diffs)):
                    print(f"  {i:2d} -> {dt} | Time diff: {td:.1f} min")

if __name__ == "__main__":
    # For testing purposes, create a dummy DatasetConfig.
    from dataclasses import dataclass
    from typing import Optional, Callable

    @dataclass
    class DummyDatasetConfig:
        sequence_length_minutes: int
        time_interval_minutes: int
        prediction_horizon: str  # Not used for multi-target here.
        transform: Optional[Callable] = None
        target_transform: Optional[Callable] = None

    # Example configuration: use 30 minutes of data (e.g., 6 bars if time_interval is 5 minutes).
    dummy_config = DummyDatasetConfig(
        sequence_length_minutes=30,
        time_interval_minutes=5,
        prediction_horizon="next_one",  # Ignored in this multi-target setup.
        transform=None,
        target_transform=None
    )

    data_file = "data/oro_transformed_enhanced.parquet"

    # Instantiate the dataset.
    dataset = CustomTradingSequenceDataset(data_file, dummy_config)
    print("Number of valid samples in dataset:", len(dataset))
    
    # Retrieve the first sample and print its shapes.
    sample_input, sample_target = dataset[0]
    print("Sample input shape:", sample_input.shape)  # Expect [sequence_length_bars, num_features]
    print("Sample target shape:", sample_target.shape)  # Expect [20]

    # Optionally, print debug information for the first sequence.
    dataset.debug_sequence_datetimes(0)
