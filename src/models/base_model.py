import math
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    Description:
        Base model class to be used with the generic Lightning wrapper.
        Provides an initialization hook and a method to configure a cosine
        annealing learning rate scheduler with a warmup phase.
    Args:
        None
    Raises:
        None
    Return:
        An instance of BaseModel.
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        self.initialize()

    def initialize(self):
        """
        Description:
            Initializes model parameters. Override this method in subclasses
            if custom initialization is needed.
        Args:
            None
        Raises:
            None
        Return:
            None
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            Forward pass method. Must be implemented in subclass.
        Args:
            x (torch.Tensor): Input tensor.
        Raises:
            NotImplementedError: If not implemented in subclass.
        Return:
            torch.Tensor: Output tensor.
        """
        raise NotImplementedError("Forward method not implemented.")

    def configure_scheduler(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
        """
        Description:
            Configures a cosine annealing learning rate scheduler with a warmup phase.
        Args:
            optimizer (torch.optim.Optimizer): Optimizer instance to attach the scheduler to.
            warmup_steps (int): Number of steps for the linear warmup.
            total_steps (int): Total number of training steps.
        Raises:
            None
        Return:
            torch.optim.lr_scheduler.LambdaLR: A learning rate scheduler instance.
        """
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler
