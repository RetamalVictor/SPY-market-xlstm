from torch.utils.data import DataLoader

from src.config import DatasetConfig
from src.dataset import CustomTradingSequenceDataset

# Example usage in main.py
if __name__ == "__main__":
    config = DatasetConfig(
        sequence_length_minutes=60,  # e.g. 1 hour of past data
        prediction_horizon='next_hour',  # 1 hour in the future
        time_interval_minutes=5
    )

    # Initialize dataset
    dataset = CustomTradingSequenceDataset(data_file="data/oro_transformed.parquet", config=config)

    # If dataset is empty, debug possible issues
    if len(dataset) == 0:
        print("No valid samples found. Possible causes:")
        print("- CSV format not matching expectation.")
        print("- Not enough data to form 1h input + 1h horizon.")
        print("- Missing or invalid datetimes.")
    else:
        # Create DataLoader with recommended performance settings
        # - 'pin_memory=True' can help if using CUDA
        # - 'num_workers' can be increased if you have multiple CPU cores
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Let's debug a single dataset index before training
        debug_index = 99990  # pick any valid dataset index
        if debug_index < len(dataset):
            dataset.debug_sequence_datetimes(debug_index)

        # Now iterate through one batch
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Inputs shape: {inputs.shape}")   # e.g., [64, 12, 6] if 12 bars x 6 features
            print(f"  Targets shape: {targets.shape}") # e.g., [64]
            break  # Just show the first batch for demonstration
