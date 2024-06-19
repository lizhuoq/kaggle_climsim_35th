from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        if configs.add_cnn:
            self.cnn = nn.Sequential(
                nn.Conv1d(configs.in_channel, configs.in_channel * 4, kernel_size=3, padding=1), 
                nn.BatchNorm1d(configs.in_channel * 4), 
                nn.GELU(), 
                nn.Conv1d(configs.in_channel * 4, configs.in_channel, kernel_size=3, padding=1), 
                nn.BatchNorm1d(configs.in_channel)
            )
            self.gelu = nn.GELU()
        else:
            self.cnn = None
        self.lstm = nn.LSTM(
            input_size=configs.in_channel, 
            hidden_size=configs.d_model, 
            num_layers=configs.n_layers, 
            batch_first=True, 
            dropout=configs.dropout, 
            bidirectional=configs.bidirectional
        )
        if configs.add_sa:
            self.attention = nn.MultiheadAttention(
                embed_dim=configs.d_model * 2 if configs.bidirectional else configs.d_model, 
                num_heads=configs.nhead, 
                dropout=configs.dropout, 
                batch_first=True
            )
        else:
            self.attention = None
        self.projection = nn.Linear(
            in_features=configs.d_model * 2 if configs.bidirectional else configs.d_model, 
            out_features=configs.out_channel
        )

    def forward(self, x):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel) 0 - 60
        Return shape:
            (batch_size, seq_len, out_channel) 0 - 60
        """
        if self.cnn is not None:
            y = self.cnn(x.transpose(1, 2)).transpose(1, 2)
            x = self.gelu(x + y)
        x, _ = self.lstm(x)
        if self.attention is not None:
            x = torch.flip(x, [1])
            # x, _ = self.attention(x, x, x, attn_mask=nn.Transformer.generate_square_subsequent_mask(60, device=x.device))
            x, _ = self.attention(x, x, x)
            x = torch.flip(x, [1])
        return self.projection(x)