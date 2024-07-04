from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        assert configs.bidirectional
        self.patch_size = configs.patch_size
        self.num_patch = 60 // self.patch_size
        self.lstm = nn.LSTM(
            input_size=configs.in_channel * self.patch_size, 
            hidden_size=configs.d_model * self.patch_size, 
            num_layers=configs.n_layers, 
            batch_first=True, 
            dropout=configs.dropout, 
            bidirectional=configs.bidirectional
        )
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Sequential(
            nn.Linear(configs.d_model * self.patch_size * 2, self.patch_size * configs.out_channel * 2), 
            nn.GELU(), 
            nn.Linear(self.patch_size * configs.out_channel * 2, self.patch_size * configs.out_channel)
        )

    def forward(self, x: torch.Tensor):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel) 0 - 60
        Return shape:
            (batch_size, seq_len, out_channel) 0 - 60
        """
        x = x.unfold(1, self.patch_size, self.patch_size).transpose(-1, -2) # b, num_patch, patch_size, in_channel
        x = x.reshape(x.shape[0], x.shape[1], -1) # b, num_patch, patch_size * in_channel
        x, _ = self.lstm(x) # b, num_patch, d_model * patch_size * 2
        x = self.dropout(x)
        x = self.projection(x) # b, num_patch, patch_size * out_channel
        x = x.reshape(x.shape[0], x.shape[1], self.patch_size, -1)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], -1)
        return x   