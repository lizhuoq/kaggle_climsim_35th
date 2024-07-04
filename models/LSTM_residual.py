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


class LSTM_Block(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model, 
            num_layers=1,
            batch_first=True, 
            dropout=dropout, 
            bidirectional=True, 
        )
        self.feedforward = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4), 
            nn.ReLU(), 
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        y, _ = self.lstm(x)
        y = self.feedforward(y)
        return y + x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.embedding = nn.Linear(configs.in_channel, configs.d_model)
        blk = [LSTM_Block(configs.d_model, configs.dropout) for _ in range(configs.n_layers)]
        self.blks = nn.Sequential(*blk)
        self.se = SELayer(configs.d_model, reduction=16)
        self.projection = nn.Linear(configs.d_model, configs.out_channel)
        
    def forward(self, x):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel) 0 - 60
        Return shape:
            (batch_size, seq_len, out_channel) 0 - 60
        """
        x = self.embedding(x)
        x = self.blks(x)
        x = self.se(x)
        return self.projection(x)