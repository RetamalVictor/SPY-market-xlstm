import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import Dataset

from src.config import DatasetConfig

class CustomTradingSequenceDataset(Dataset):
    def __init__(self, data_file: str, config: 'DatasetConfig'):
        """
        Description: Handles creation of a time-series dataset for trading bar data.
        args:
            data_file (str): Path to CSV or Parquet file containing trading data.
            config (DatasetConfig): Configuration object with dataset settings.
        raises:
            ValueError: If prediction_horizon is invalid or not enough rows for configuration.
        return: 
            CustomTradingSequenceDataset instance.
        """
        self.config = config
        self.transform = config.transform
        self.target_transform = config.target_transform

        #############################
        # 1. Load the CSV
        #############################
        # - header=0: first row is column names
        # - memory_map=True can help for large CSVs on some systems
        # - dtype: Convert columns directly to float/int
        # Ensure your CSV has the columns:
        # DTYYYYMMDD,TIME,OPEN,HIGH,LOW,CLOSE,VOL,OPENINT
        print(f"Loading preprocessed dataset from {data_file}...")
        
        # Check if it's a Parquet file
        if data_file.endswith(".parquet"):
            df = pd.read_parquet(data_file)
        else:
            df = pd.read_csv(data_file, memory_map=True)

        initial_len = len(df)
        print(f"Read {initial_len} rows from CSV.")

        #############################
        # 2. Parse and Clean Datetime
        #############################
        # Format "%Y%m%d%H%M%S" => e.g. 20061204 + 000500 => "20061204000500"
        df["DATETIME"] = pd.to_datetime(
            df["DTYYYYMMDD"] + df["TIME"],
            format="%Y%m%d%H%M%S",
            errors="coerce"
        )
        # Drop rows where DATETIME couldn't be parsed
        df.dropna(subset=["DATETIME"], inplace=True)
        df.sort_values(by="DATETIME", inplace=True)
        df.reset_index(drop=True, inplace=True)
        after_drop_len = len(df)
        print(f"Dropped {initial_len - after_drop_len} rows due to invalid datetime parsing.")

        #############################
        # 3. Convert Config Times
        #############################
        # Convert 'sequence_length_minutes' -> bars
        self.sequence_length_bars = config.sequence_length_minutes // config.time_interval_minutes

        # Convert 'prediction_horizon' -> total horizon in minutes -> bars
        if config.prediction_horizon == 'next_half_hour':
            horizon_minutes = 30
        elif config.prediction_horizon == 'next_hour':
            horizon_minutes = 60
        elif config.prediction_horizon == 'next_day':
            horizon_minutes = 1440  # 24 hours
        else:
            raise ValueError("Invalid prediction_horizon. Choose from 'next_half_hour','next_hour','next_day'.")

        self.prediction_horizon_bars = horizon_minutes // config.time_interval_minutes

        #############################
        # 4. Prepare Features & Datetimes
        #############################
        self.features = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOL", "OPENINT", "TIME_DIFF"]].values
        self.datetimes = df["DATETIME"].values

        total_bars = len(self.features)
        print(f"Total bars (rows) after cleaning: {total_bars}")

        #############################
        # 5. Find Valid Indices
        #############################
        # We define a valid index if:
        #   - The end of the input sequence fits inside the dataset
        #   - The horizon bar also fits
        # E.g., if start_idx=0, we use [0..sequence_length_bars-1] for input,
        # and then target is at (start_idx + sequence_length_bars + prediction_horizon_bars - 1).
        self.valid_indices = []
        for start_idx in range(total_bars):
            end_of_input = start_idx + self.sequence_length_bars - 1
            target_idx = end_of_input + self.prediction_horizon_bars
            if target_idx < total_bars:
                self.valid_indices.append(start_idx)

        self.total_samples = len(self.valid_indices)
        print(f"Initialized dataset with {self.total_samples} valid samples.")

    def __len__(self) -> int:
        return self.total_samples


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a valid sequence and its corresponding target.

        - Ensures that all time differences in the sequence are ≤ 5 minutes.
        - If not, it tries the next index until a valid one is found.
        - Raises an IndexError if no valid sequence is found.

        Args:
            idx (int): The index of the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (input_sequence, target)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        while idx < len(self):
            start_idx = self.valid_indices[idx]
            end_of_input = start_idx + self.sequence_length_bars
            target_idx = end_of_input + self.prediction_horizon_bars - 1
            time_diffs = self.features[start_idx:end_of_input, 6]

            if (time_diffs <= 5).all():
                input_seq = self.features[start_idx:end_of_input]
                input_seq = torch.tensor(input_seq, dtype=torch.float32)
                target = self.features[target_idx, 3]
                target = torch.tensor(target, dtype=torch.float32)

                if self.transform:
                    input_seq = self.transform(input_seq)
                if self.target_transform:
                    target = self.target_transform(target)
                return input_seq, target

            idx += 1

        raise IndexError("No valid sequence found at or after the given index.")

    #############################
    # 6. Debug Methods
    #############################
    def debug_sequence_datetimes(self, idx: int):
        """
        Print the DATETIME and TIME_DIFF for each bar in the input sequence at dataset index `idx`.
        This helps confirm we're fetching the correct times and computed time differences.

        Args:
            idx (int): Index of the dataset sample to debug.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        start_idx = self.valid_indices[idx]
        end_of_input = start_idx + self.sequence_length_bars

        seq_datetimes = self.datetimes[start_idx : end_of_input]
        time_diffs = self.features[start_idx : end_of_input, -1]  # Last column contains TIME_DIFF

        print(f"\n[DEBUG] Sequence datetimes & time differences for dataset index {idx}:")
        for i, (dt, td) in enumerate(zip(seq_datetimes, time_diffs)):
            print(f"  {i:2d} -> {dt} | Time diff: {td:.1f} min")

    def find_sequences_with_large_time_diff(self, threshold: float = 5.0):
        """
        Iterates through all possible sequences and prints those with any TIME_DIFF > threshold.
        """
        print(f"\nScanning for sequences with any TIME_DIFF > {threshold} minutes...")
        total_bars = len(self.features)
        limit = total_bars - self.sequence_length_bars - self.prediction_horizon_bars + 1

        for start_idx in range(limit):
            time_diffs = self.features[start_idx:start_idx + self.sequence_length_bars, 6]
            if (time_diffs > threshold).any():
                seq_datetimes = self.datetimes[start_idx:start_idx + self.sequence_length_bars]
                print(f"\nSequence starting at index {start_idx}:")
                for i, (dt, td) in enumerate(zip(seq_datetimes, time_diffs)):
                    print(f"  {i:2d} -> {dt} | Time diff: {td:.1f} min")
