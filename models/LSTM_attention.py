from torch import nn
import torch
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # TODO: add CNN
        self.lstm = nn.LSTM(
            input_size=configs.in_channel, 
            hidden_size=configs.d_model, 
            num_layers=configs.n_layers, 
            batch_first=True, 
            dropout=configs.dropout, 
            bidirectional=configs.bidirectional
        )
        # TODO: add self-attention
        self.projection = nn.Linear(
            in_features=configs.d_model * 2 if configs.bidirectional else configs.d_model, 
            out_features=configs.out_channel
        )

    def forward(self, x: torch.Tensor):
        """
        Input shape:
            x: (batch_size, range(60), in_channel)
        Return shape:
            (batch_size, range(60), out_channel)
        """
        x = torch.flip(x, [1])
        x, _ = self.lstm(x)
        x = self.projection(x)
        return torch.flip(x, [1])