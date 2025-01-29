import pandas as pd

def preprocess_and_save(input_file: str, output_file: str, save_as_parquet: bool = True):
    """
    Preprocess trading data to add time differences and detect trading gaps.

    Args:
        input_file (str): Path to the raw CSV file.
        output_file (str): Path to save the processed dataset.
        save_as_parquet (bool): If True, saves as Parquet; otherwise, saves as CSV.
    """
    print(f"Reading raw dataset from {input_file}...")
    
    # Load the dataset
    df = pd.read_csv(
        input_file,
        delimiter=",",
        header=0,
        dtype={
            "DTYYYYMMDD": str,
            "TIME": str,
            "OPEN": float,
            "HIGH": float,
            "LOW": float,
            "CLOSE": float,
            "VOL": int,
            "OPENINT": int
        }
    )

    # Parse datetime
    df["DATETIME"] = pd.to_datetime(
        df["DTYYYYMMDD"] + df["TIME"],
        format="%Y%m%d%H%M%S",
        errors="coerce"
    )

    # Drop invalid rows
    df.dropna(subset=["DATETIME"], inplace=True)
    df.sort_values(by="DATETIME", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Extract Date (Trading Days) and ensure correct type
    df["DATE_ONLY"] = df["DATETIME"].dt.date  # Convert to datetime.date format

    print(f"Dataset contains {len(df)} rows after cleaning.")

    # Compute time differences (Ensure proper fillna without inplace=True)
    df["TIME_DIFF"] = df["DATETIME"].diff().dt.total_seconds() / 60.0  # Convert to minutes
    df["TIME_DIFF"] = df["TIME_DIFF"].fillna(0)  # Assign explicitly to avoid inplace warning

    # Detect **New Trading Days** (When `DATE_ONLY` changes)
    df["NEW_TRADING_DAY"] = df["DATE_ONLY"] != df["DATE_ONLY"].shift(1)
    df.loc[df["NEW_TRADING_DAY"], "TIME_DIFF"] = None  # Reset time diff on new trading day

    # Detect **Large Gaps** (e.g., Overnight or Multi-Day Gaps)
    df["LARGE_GAP"] = df["TIME_DIFF"] > 120  # Flag gaps > 2 hours

    # Ensure correct dtype for DATE_ONLY before saving
    df["DATE_ONLY"] = pd.to_datetime(df["DATE_ONLY"])  # Converts to datetime type

    # Save processed dataset
    if save_as_parquet:
        output_file += ".parquet"
        df.to_parquet(output_file, index=False, engine="pyarrow")  # Explicit engine
        print(f"Saved processed dataset as Parquet: {output_file}")
    else:
        output_file += ".csv"
        df.to_csv(output_file, index=False)
        print(f"Saved processed dataset as CSV: {output_file}")

if __name__ == "__main__":
    # Example usage: Process and save as Parquet
    preprocess_and_save("data/oro5min.txt", "data/oro_transformed_gaps", save_as_parquet=True)
