from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.embedding = nn.Linear(16, configs.d_model)
        self.lstm = nn.LSTM(
            input_size=configs.in_channel - 16, 
            hidden_size=configs.d_model, 
            num_layers=configs.n_layers, 
            batch_first=True, 
            dropout=configs.dropout, 
            bidirectional=configs.bidirectional
        )
        self.projection = nn.Linear(
            in_features=configs.d_model * 2 if configs.bidirectional else configs.d_model, 
            out_features=configs.out_channel
        )
        self.configs = configs

    def forward(self, x: torch.Tensor):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel)
        Return shape:
            (batch_size, seq_len, out_channel)
        """
        init = x[:, 0, 9:]
        init: torch.Tensor = self.embedding(init)
        if self.configs.bidirectional:
            init = init.unsqueeze(0).repeat(2 * self.configs.n_layers, 1, 1)
        else:
            init = init.unsqueeze(0).repeat(self.configs.n_layers, 1, 1)
        x = x[:, :, :9]
        x, _ = self.lstm(x, (init, init))
        return self.projection(x)