import torch

class TimeDifferenceTransform:
    def __call__(self, sequence: torch.Tensor, datetimes: list) -> torch.Tensor:
        """
        Compute time difference (in minutes) between entries in the sequence.
        
        Args:
            sequence (torch.Tensor): Tensor of shape (sequence_length, num_features).
            datetimes (list): List of datetime objects corresponding to the sequence.

        Returns:
            torch.Tensor: Tensor with time differences added as an extra feature.
        """
        sequence_length, num_features = sequence.shape

        # Compute time differences in minutes
        time_diffs = torch.zeros(sequence_length, 1, dtype=torch.float32)  # Initialize with zeros

        for i in range(1, sequence_length):
            time_diffs[i] = (datetimes[i] - datetimes[i - 1]).total_seconds() / 60.0  # Convert seconds to minutes

        # Concatenate the new time_diff feature to the original feature set
        return torch.cat([sequence, time_diffs], dim=1)  # Shape: (sequence_length, num_features + 1)
