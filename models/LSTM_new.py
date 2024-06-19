from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.global_embedding = nn.Linear(configs.in_channel - 9, configs.d_model)
        self.level_embedding = nn.Linear(9, configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.lstm = nn.LSTM(
            input_size=configs.d_model, 
            hidden_size=configs.d_model, 
            num_layers=configs.n_layers, 
            batch_first=True, 
            dropout=configs.dropout, 
            bidirectional=configs.bidirectional
        )
        out_channel = configs.d_model * 2 if configs.bidirectional else configs.d_model
        self.global_projection = nn.Linear(out_channel * 2, 8)
        self.level_projection = nn.Linear(out_channel, 6)

    def forward(self, x):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel)
        Return shape:
            (batch_size, seq_len, out_channel)
        """
        global_x = x[:, 0, 9:]
        level_x = x[:, :, :9]
        level_x = self.level_embedding(level_x)
        global_x = self.global_embedding(global_x).unsqueeze(1)
        x = torch.cat([global_x, level_x, global_x], dim=1)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        global_out = self.global_projection(torch.cat([x[:, 0, :], x[:, -1, :]], dim=1)).unsqueeze(1).repeat(1, 60, 1)
        level_out = self.level_projection(x[:, 1:-1, :])
        out = torch.cat([level_out, global_out], dim=2)
        return out