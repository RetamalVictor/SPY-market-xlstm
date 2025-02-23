import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from src.models.base_model import BaseModel

###############################################
# Helper Modules
###############################################

class CausalConv1D(nn.Module):
    """
    1D causal convolution. The output is shifted so that it does not depend on future time steps.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, seq_len]
        x = self.conv(x)
        if self.padding:
            return x[:, :, :-self.padding]
        return x

class BlockDiagonal(nn.Module):
    """
    Applies a block-diagonal linear transformation using exactly 2 blocks.
    Assumes that in_features and out_features are divisible by 2.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        assert in_features % 2 == 0, "in_features must be divisible by 2"
        assert out_features % 2 == 0, "out_features must be divisible by 2"
        block_in = in_features // 2
        block_out = out_features // 2
        self.block1 = nn.Linear(block_in, block_out)
        self.block2 = nn.Linear(block_in, block_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_features]
        split1, split2 = torch.chunk(x, 2, dim=-1)
        out1 = self.block1(split1)
        out2 = self.block2(split2)
        return torch.cat((out1, out2), dim=-1)

###############################################
# sLSTM Block (Scalar LSTM)
###############################################

class sLSTMBlock(nn.Module):
    """
    A block implementing the sLSTM update with exponential gating and a residual projection.
    """
    def __init__(self, input_size: int, hidden_size: int, num_heads: int, proj_factor: float = 4/3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads  # we assume this is 2 in our hardcoded architecture.
        self.proj_factor = proj_factor

        self.layer_norm = nn.LayerNorm(input_size)
        self.causal_conv = CausalConv1D(in_channels=1, out_channels=1, kernel_size=4)

        # Use BlockDiagonal (hardcoded for 2 blocks)
        self.Wz = BlockDiagonal(input_size, hidden_size)
        self.Wi = BlockDiagonal(input_size, hidden_size)
        self.Wf = BlockDiagonal(input_size, hidden_size)
        self.Wo = BlockDiagonal(input_size, hidden_size)
        self.Rz = BlockDiagonal(hidden_size, hidden_size)
        self.Ri = BlockDiagonal(hidden_size, hidden_size)
        self.Rf = BlockDiagonal(hidden_size, hidden_size)
        self.Ro = BlockDiagonal(hidden_size, hidden_size)

        # Use 2 groups for GroupNorm.
        self.group_norm = nn.GroupNorm(2, hidden_size)

        self.up_proj_left = nn.Linear(hidden_size, int(hidden_size * proj_factor))
        self.up_proj_right = nn.Linear(hidden_size, int(hidden_size * proj_factor))
        self.down_proj = nn.Linear(int(hidden_size * proj_factor), input_size)

    def forward(
        self, 
        x: torch.Tensor,
        prev_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # x: [batch, input_size]
        # prev_state: (h_prev, c_prev, n_prev, m_prev), each [batch, hidden_size]
        h_prev, c_prev, n_prev, m_prev = prev_state

        x_norm = self.layer_norm(x)
        x_conv = F.silu(self.causal_conv(x_norm.unsqueeze(1)).squeeze(1))

        z = torch.tanh(self.Wz(x) + self.Rz(h_prev))
        o = torch.sigmoid(self.Wo(x) + self.Ro(h_prev))
        i_tilde = self.Wi(x_conv) + self.Ri(h_prev)
        f_tilde = self.Wf(x_conv) + self.Rf(h_prev)

        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)

        c_t = f * c_prev + i * z
        n_t = f * n_prev + i

        h_t = o * (c_t / n_t)

        output = h_t
        output_norm = self.group_norm(output)
        output_left = self.up_proj_left(output_norm)
        output_right = self.up_proj_right(output_norm)
        output_gated = F.gelu(output_right)
        output = output_left * output_gated
        output = self.down_proj(output)
        final_output = output + x

        return final_output, (h_t, c_t, n_t, m_t)


###############################################
# mLSTM Block (Matrix LSTM)
###############################################

class mLSTMBlock(nn.Module):
    """
    A block implementing the mLSTM update with matrix-style operations and an attention-inspired mechanism.
    """
    def __init__(self, input_size: int, hidden_size: int, num_heads: int, proj_factor: float = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads  # assume 2
        self.proj_factor = proj_factor

        self.layer_norm = nn.LayerNorm(input_size)
        self.up_proj_left = nn.Linear(input_size, int(input_size * proj_factor))
        self.up_proj_right = nn.Linear(input_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, input_size)

        self.causal_conv = CausalConv1D(in_channels=1, out_channels=1, kernel_size=4)
        self.skip_connection = nn.Linear(int(input_size * proj_factor), hidden_size)

        self.Wq = BlockDiagonal(int(input_size * proj_factor), hidden_size)
        self.Wk = BlockDiagonal(int(input_size * proj_factor), hidden_size)
        self.Wv = BlockDiagonal(int(input_size * proj_factor), hidden_size)
        self.Wi = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wf = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wo = nn.Linear(int(input_size * proj_factor), hidden_size)

        self.group_norm = nn.GroupNorm(2, hidden_size)

    def forward(
        self, 
        x: torch.Tensor,
        prev_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # x: [batch, input_size]
        # prev_state: (h_prev, c_prev, n_prev, m_prev), each [batch, hidden_size]
        h_prev, c_prev, n_prev, m_prev = prev_state

        x_norm = self.layer_norm(x)
        x_up_left = self.up_proj_left(x_norm)
        x_up_right = self.up_proj_right(x_norm)

        x_conv = F.silu(self.causal_conv(x_up_left.unsqueeze(1)).squeeze(1))
        x_skip = self.skip_connection(x_conv)

        q = self.Wq(x_conv)
        head_size = self.hidden_size // self.num_heads
        k = self.Wk(x_conv) / math.sqrt(head_size)
        v = self.Wv(x_up_left)

        i_tilde = self.Wi(x_conv)
        f_tilde = self.Wf(x_conv)
        o = torch.sigmoid(self.Wo(x_up_left))

        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)

        c_t = f * c_prev + i * (v * k)
        n_t = f * n_prev + i * k

        dot = torch.sum(n_t * q, dim=-1, keepdim=True)
        denom = torch.clamp(torch.abs(dot), min=1.0)
        h_t = o * ((c_t * q) / denom)

        output = h_t
        output_norm = self.group_norm(output)
        output = output_norm + x_skip
        output = output * F.silu(x_up_right)
        output = self.down_proj(output)
        final_output = output + x

        return final_output, (h_t, c_t, n_t, m_t)


###############################################
# xLSTM Model with Hardcoded Layers: ["s", "m", "s"]
###############################################

LayerState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

class xLSTM(BaseModel):
    """
    Extended LSTM (xLSTM) with hardcoded layers: sLSTMBlock, mLSTMBlock, sLSTMBlock.
    
    Args:
        input_size (int): Dimensionality of the input.
        hidden_size (int): Dimensionality of the hidden state.
        num_heads (int): Must be 2 in this hardcoded version.
        batch_first (bool): If True, input/output tensors are of shape (batch, seq, features).
        proj_factor_slstm (float): Projection factor for sLSTM blocks.
        proj_factor_mlstm (float): Projection factor for mLSTM blocks.
    """
    def __init__(self, input_size: int, hidden_size: int, num_heads: int,
                 batch_first: bool = False, proj_factor_slstm: float = 4/3, proj_factor_mlstm: float = 2):
        super().__init__()
        assert num_heads == 2, "In the hardcoded version, num_heads must be 2."
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads  # Will be 2.
        self.batch_first = batch_first
        self.proj_factor_slstm = proj_factor_slstm
        self.proj_factor_mlstm = proj_factor_mlstm

        # Hardcode three layers: sLSTMBlock, mLSTMBlock, sLSTMBlock.
        self.block1 = sLSTMBlock(input_size, hidden_size, num_heads, proj_factor_slstm)
        self.block2 = mLSTMBlock(input_size, hidden_size, num_heads, proj_factor_mlstm)
        self.block3 = sLSTMBlock(input_size, hidden_size, num_heads, proj_factor_slstm)
        self.num_layers = 3

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[LayerState, LayerState, LayerState]] = None
    ) -> Tuple[torch.Tensor, Tuple[LayerState, LayerState, LayerState]]:
        """
        Description:
            Forward pass through the 3-layer xLSTM (s-m-s).
        Args:
            x (torch.Tensor): [batch, seq, input_size] if batch_first=True, else [seq, batch, input_size].
            state (tuple): Optional tuple of 3 layer states, each a tuple (h, c, n, m).
        Return:
            outputs (torch.Tensor): [batch, seq, input_size] if batch_first=True, else [seq, batch, input_size].
            state (tuple): The updated 3 layer states.
        """
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        if state is None:
            s1 = (
                torch.zeros(batch_size, self.hidden_size, device=x.device),
                torch.zeros(batch_size, self.hidden_size, device=x.device),
                torch.zeros(batch_size, self.hidden_size, device=x.device),
                torch.zeros(batch_size, self.hidden_size, device=x.device)
            )
            s2 = (
                torch.zeros(batch_size, self.hidden_size, device=x.device),
                torch.zeros(batch_size, self.hidden_size, device=x.device),
                torch.zeros(batch_size, self.hidden_size, device=x.device),
                torch.zeros(batch_size, self.hidden_size, device=x.device)
            )
            s3 = (
                torch.zeros(batch_size, self.hidden_size, device=x.device),
                torch.zeros(batch_size, self.hidden_size, device=x.device),
                torch.zeros(batch_size, self.hidden_size, device=x.device),
                torch.zeros(batch_size, self.hidden_size, device=x.device)
            )
            state = (s1, s2, s3)

        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            s1, s2, s3 = state
            x_t, new_s1 = self.block1(x_t, s1)
            x_t, new_s2 = self.block2(x_t, s2)
            x_t, new_s3 = self.block3(x_t, s3)
            state = (new_s1, new_s2, new_s3)
            outputs.append(x_t)

        outputs = torch.stack(outputs)  # [seq, batch, input_size]
        if self.batch_first:
            outputs = outputs.transpose(0, 1)

        return outputs, state

    @classmethod
    def from_config(cls, input_size: int, config) -> "xLSTM":
        """
        Instantiates xLSTM using parameters from the model-specific configuration.
        Args:
            input_size (int): The number of input features.
            config: Configuration object with at least:
                    - hidden_size
                    - num_heads (must be 2)
                    Optionally:
                    - batch_first (bool)
                    - proj_factor_slstm (float)
                    - proj_factor_mlstm (float)
        Returns:
            An instance of xLSTM.
        """
        return cls(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,  # must be 2
            batch_first=getattr(config, 'batch_first', False),
            proj_factor_slstm=getattr(config, 'proj_factor_slstm', 4/3),
            proj_factor_mlstm=getattr(config, 'proj_factor_mlstm', 2)
        )


###############################################
# xLSTMWrapper with final projection
###############################################

class xLSTMWrapper(BaseModel):
    """
    Wraps an xLSTM module and applies a final linear projection so that the final output
    has a desired output dimension. If you are doing binary classification, set output_size=1
    so that it produces a single logit.
    """
    def __init__(self, input_size: int, hidden_size: int, num_heads: int, output_size: int,
                 batch_first: bool = False, proj_factor_slstm: float = 4/3, proj_factor_mlstm: float = 2):
        super().__init__()
        self.xlstm = xLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=num_heads,  # must be 2
            batch_first=batch_first,
            proj_factor_slstm=proj_factor_slstm,
            proj_factor_mlstm=proj_factor_mlstm
        )
        # For binary classification, set output_size = 1
        self.final_projection = nn.Linear(input_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[LayerState, LayerState, LayerState]] = None
    ) -> Tuple[torch.Tensor, Tuple[LayerState, LayerState, LayerState]]:
        """
        Description:
            Forward pass of the xLSTMWrapper. First processes x through xLSTM, then
            applies a final linear projection for either regression or classification.
        Args:
            x (torch.Tensor): [batch, seq, input_size] if batch_first=True, else [seq, batch, input_size].
            state (tuple): Optional states for the 3-layer xLSTM.
        Returns:
            outputs (torch.Tensor): [batch, seq, output_size] if batch_first=True.
            state (tuple): The updated 3-layer xLSTM states.
        """
        outputs, state = self.xlstm(x, state)
        outputs = self.final_projection(outputs)
        return outputs, state

    @classmethod
    def from_config(cls, input_size: int, output_size: int, config) -> "xLSTMWrapper":
        """
        Instantiates xLSTMWrapper using parameters from the model-specific configuration.
        Args:
            input_size (int): The number of input features.
            output_size (int): The desired output dimension. For binary classification, use 1.
            config: Configuration object with at least:
                    - hidden_size
                    - num_heads (must be 2)
                    Optionally:
                    - batch_first (bool)
                    - proj_factor_slstm (float)
                    - proj_factor_mlstm (float)
        Returns:
            An instance of xLSTMWrapper.
        """
        return cls(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            output_size=output_size,
            batch_first=getattr(config, 'batch_first', False),
            proj_factor_slstm=getattr(config, 'proj_factor_slstm', 4/3),
            proj_factor_mlstm=getattr(config, 'proj_factor_mlstm', 2)
        )

    def configure_scheduler(self, optimizer, warmup_steps, total_steps):
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Example Usage
if __name__ == "__main__":
    class XLSTMConfig:
        hidden_size = 256
        num_heads = 2  # Must be 2 in this hardcoded version.
        batch_first = True
        proj_factor_slstm = 4/3
        proj_factor_mlstm = 2

    config = XLSTMConfig()
    input_size = 64

    # For binary classification, set output_size = 1
    output_size = 1

    model_wrapper = xLSTMWrapper.from_config(input_size=input_size, output_size=output_size, config=config)
    print(model_wrapper)

    # Example input: [batch=2, seq=10, features=64]
    example_input = torch.randn(2, 10, 64)
    outputs, _ = model_wrapper(example_input)
    print("Output shape:", outputs.shape)  # Expect [2, 10, 1] for binary classification
