import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate time series data
n_points = 1000
x = np.linspace(0, 4 * np.pi, n_points)
cos_series = np.cos(x) + np.random.normal(0, 0.1, n_points)
sin_series = np.sin(x) + np.random.normal(0, 0.2, n_points)
tan_series = np.tan(x) + np.random.normal(0, 0.5, n_points)
combined_series = np.cos(x) + np.sin(x) + np.random.normal(0, 0.3, n_points)

# Clip the tan series for better visualization
tan_series = np.clip(tan_series, -10, 10)

# Create correlated series
cos_series_correlated = cos_series * 1.2 + np.random.normal(0, 0.05, n_points)
combined_series_correlated = combined_series * 0.8 + np.random.normal(0, 0.1, n_points)

# Add low-frequency disturbances
low_freq_disturbance = np.sin(x / 5) * 0.5
cos_series_correlated_disturbed = cos_series_correlated + low_freq_disturbance
combined_series_correlated_disturbed = combined_series_correlated + low_freq_disturbance

# Compile all series into a DataFrame
df = pd.DataFrame({
    'x': x,
    'cos_series': cos_series,
    'sin_series': sin_series,
    'tan_series': tan_series,
    'combined_series': combined_series,
    'cos_series_correlated': cos_series_correlated,
    'combined_series_correlated': combined_series_correlated,
    'cos_series_correlated_disturbed': cos_series_correlated_disturbed,
    'combined_series_correlated_disturbed': combined_series_correlated_disturbed,
    'low_freq_disturbance': low_freq_disturbance
})

# Save the DataFrame to a CSV file
csv_file_path = 'time_series_data_2.csv'
df.to_csv(csv_file_path, index=False)

# Read the CSV file and plot all the series
df_read = pd.read_csv(csv_file_path)

plt.figure(figsize=(14, 12))

for i, column in enumerate(df_read.columns[1:], 1):  # Skip the 'x' column for plotting
    plt.subplot(5, 2, i)
    plt.plot(df_read['x'], df_read[column], label=column)
    plt.title(column)
    plt.legend()

plt.tight_layout()
plt.show()


