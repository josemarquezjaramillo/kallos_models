import torch
import torch.nn as nn

class DirectionSelectiveMSELoss(nn.Module):
    def __init__(self, direction_penalty=5.0):
        super().__init__()
        self.direction_penalty = direction_penalty

    def forward(self, y_hat, y_true):
        # y_hat and y_true: shape (batch, seq) or (batch,)
        # direction_indicator: 1 if same sign, 0 if not
        direction_indicator = (y_hat * y_true > 0).float()
        
        # Standard MSE
        mse = (y_hat - y_true) ** 2
        
        # Only penalize where direction_indicator is 0 (wrong direction)
        adjusted_mse = mse * (direction_indicator + self.direction_penalty * (1 - direction_indicator))
        
        return adjusted_mse.mean()
