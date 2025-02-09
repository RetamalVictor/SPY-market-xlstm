import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Description:
        A simple LSTM-based model for time-series prediction. The model processes an input sequence
        using one or more LSTM layers and outputs a prediction based on the last time step's hidden state.
    
    Args:
        input_size (int): Number of features in the input.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int): Number of LSTM layers.
        output_size (int): Number of outputs.
        dropout (float, optional): Dropout probability applied between LSTM layers (if num_layers > 1). Default is 0.0.
    
    Raises:
        None
    
    Returns:
        LSTMModel instance.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    @classmethod
    def from_config(cls, input_size: int, output_size:int,  config):
        """
        Description:
            Instantiates SimpleNN using parameters from the model-specific configuration.
        Args:
            input_size (int): The number of input features.
            config (SimpleNNConfig): Model-specific configuration (e.g., hidden_size).
        Returns:
            An instance of SimpleNN.
        """
        return cls(input_size=input_size, 
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    dropout=config.dropout, 
                    output_size=output_size
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            Forward pass for the LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        
        Raises:
            None
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size), where the prediction is based on
                          the last time step's hidden state.
        """
        # Initialize hidden and cell states with zeros.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        # Pass through LSTM.
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, seq_length, hidden_size)
        
        # Select the output corresponding to the last time step.
        last_output = out[:, -1, :]  # shape: (batch_size, hidden_size)
        
        # Pass through the fully connected layer to obtain the final output.
        output = self.relu(self.fc(last_output))
        output = self.fc2(output)
        return output
    
