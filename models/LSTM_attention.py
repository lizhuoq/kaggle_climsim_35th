from torch import nn
import torch


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """
        Input:
            x shape: batch_size, seq_len, c
        Return:
            shape: batch_size, seq_len, c
        """
        b, _, c = x.size()
        x = x.transpose(1, 2)
        y = self.avg_pool(x).squeeze(-1)
        y = self.fc(y).unsqueeze(-1)
        out = x * y
        return out.transpose(1, 2)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.se = SELayer(configs.in_channel, 4)
        self.lstm = nn.LSTM(
            input_size=configs.in_channel, 
            hidden_size=configs.d_model, 
            num_layers=configs.n_layers, 
            batch_first=True, 
            dropout=configs.dropout, 
            bidirectional=configs.bidirectional
        )
        # self.se = SELayer(configs.d_model * 2 if configs.bidirectional else configs.d_model, 16)
        self.projection = nn.Linear(
            configs.d_model * 2 if configs.bidirectional else configs.d_model, 
            configs.out_channel
        )

    def forward(self, x: torch.Tensor):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel)
        Return shape:
            (batch_size, seq_len, out_channel)
        """
        x = self.se(x)
        x, _ = self.lstm(x)
        x = self.projection(x)
        return x