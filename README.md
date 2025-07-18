# Trading Model Training Framework

This repository contains a PyTorch Lightning-based training framework for high-frequency trading data. It uses a YAML configuration file to define hyperparameters (including those for data processing) and logs experiment details to TensorBoard. The project includes:

- A generic Lightning module wrapper with cosine warmup for the learning rate.
- A custom dataset loader for trading sequence data.
- Inference callbacks that log predictions vs. real targets to TensorBoard.
- Example models (a simple neural network in this case).

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── training.py                # Main training script
├── inference.py               # Inference script
└── config/
    └── train_config.yaml      # YAML file with training and dataset hyperparameters
└── src/
    ├── __init__.py
    ├── config.py              # Configuration loader using dataclasses and YAML
    ├── dataset.py             # CustomTradingSequenceDataset for data handling
    ├── callbacks/
    │   ├── __init__.py
    │   └── inference_callback.py   # Callback to plot predictions vs targets
    └── models/
        ├── __init__.py
        ├── base_model.py           # Base model with cosine warmup helper
        ├── baseline.py             # An example model (SimpleNN) that extends BaseModel
        └── lightning_wrapper.py   # Generic Lightning wrapper for torch models
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/RetamalVictor/Clase_Walter.git
   cd Clase_Walter
   ```

2. **Create a Virtual Environment:**

   On Linux/Mac:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

   On Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install the Package in Editable Mode:**

   ```bash
   pip install -e .
   ```

   This installs the package using `setup.py` so that you can run the provided console scripts.

## Usage

The project defines two console scripts via `setup.py`:

- **Training:**  
  Run the training process by specifying the path to your YAML configuration file.

  ```bash
  train --config train_config.yaml
  ```

  This command will:
  - Load the configuration from `train_config.yaml`.
  - Set up the dataset and split it into training/validation sets.
  - Instantiate the model (e.g., SimpleNN wrapped with the generic Lightning module).
  - Log hyperparameters and the learning rate schedule (with cosine warmup) to TensorBoard.
  - Save the best performing model checkpoint based on validation loss.

- **Inference:**  
  Run inference using a saved checkpoint and a data file. For example:

  ```bash
  infer --checkpoint path/to/checkpoint.ckpt --data_file data/oro_transformed.parquet --evaluate
  ```

  - The `--evaluate` flag plots real targets vs. predicted values.
  - Without `--evaluate`, the script prints predictions to the console.

## TensorBoard

To monitor training metrics, hyperparameters, and the learning rate schedule, launch TensorBoard by running:

```bash
tensorboard --logdir tb_logs
```

Then open the provided URL in your browser.

## Additional Notes

- **Configuration:**  
  All experiment hyperparameters (training and dataset-related) are defined in `train_config.yaml`. This file is loaded and merged by the project, and the complete configuration is logged in TensorBoard for full reproducibility.

- **Callbacks:**  
  The framework uses several callbacks:
  - **InferencePlotCallback:** Logs a plot of predictions vs. real targets after validation.
  - **ModelCheckpoint:** Saves the best model based on `val_loss`.
  - **LearningRateMonitor:** Logs the learning rate at each step to verify the cosine warmup and annealing schedule.
```