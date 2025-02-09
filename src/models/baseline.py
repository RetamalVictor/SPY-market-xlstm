import torch
import torch.nn as nn
from src.models.base_model import BaseModel

class SimpleNN(BaseModel):
    """
    Description:
        A simple feed-forward neural network that extends BaseModel.
        This network consists of a linear layer, a ReLU activation, and an output layer.
    Args:
        input_size (int): Flattened input size.
        hidden_size (int): Number of hidden units.
        output_size (int): Number of output units.
    Raises:
        None
    Return:
        An instance of SimpleNN.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    @classmethod
    def from_config(cls, input_size: int, config):
        """
        Description:
            Instantiates SimpleNN using parameters from the model-specific configuration.
        Args:
            input_size (int): The number of input features.
            config (SimpleNNConfig): Model-specific configuration (e.g., hidden_size).
        Returns:
            An instance of SimpleNN.
        """
        return cls(input_size=input_size, hidden_size=config.hidden_size, output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            Forward pass of the simple neural network.
            Flattens the input and applies two linear layers with a ReLU.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, sequence_length, features).
        Return:
            torch.Tensor: Output tensor.
        """
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def initialize(self):
        """
        Description:
            Initializes the weights of the network using Kaiming normal initialization.
        Args:
            None
        Raises:
            None
        Return:
            None
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
