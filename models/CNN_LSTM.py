from torch import nn
import torch
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(configs.in_channel, configs.conv_out_channel, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.Conv1d(configs.conv_out_channel, configs.conv_out_channel, kernel_size=3, padding=1), 
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=configs.conv_out_channel, 
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

    def forward(self, x):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel)
        Return shape:
            (batch_size, seq_len, out_channel)
        """
        x = self.cnn(x.transpose(1, 2))
        x = F.interpolate(x, size=60, mode="linear", align_corners=False).transpose(1, 2)
        x, _ = self.lstm(x)
        return self.projection(x)