from torch import nn
import torch


class LSTM_block(nn.Module):
    def __init__(self, in_channel, d_model, out_channel, n_layers, dropout, bidirectional) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channel, 
            hidden_size=d_model, 
            num_layers=n_layers, 
            batch_first=True, 
            dropout=dropout, 
            bidirectional=bidirectional
        )
        self.projection = nn.Linear(
            in_features=d_model * 2 if bidirectional else d_model, 
            out_features=out_channel
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.projection(x)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.W = nn.Linear(configs.in_channel, configs.d_model)
        embed_dim = configs.d_model // 7
        self.embed_dim = embed_dim
        self.blk = nn.ModuleList()
        for i in range(6):
            self.blk.append(LSTM_block(embed_dim, embed_dim, 1, configs.n_layers, configs.dropout, configs.bidirectional))
        self.blk.append(LSTM_block(embed_dim, embed_dim, 8, configs.n_layers, configs.dropout, configs.bidirectional))

    def forward(self, x):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel)
        Return shape:
            (batch_size, seq_len, out_channel)
        """
        x = self.W(x)
        output = []
        for i in range(7):
            output.append(self.blk[i](x[:, :, i * self.embed_dim: (i + 1) * self.embed_dim]))
        return torch.cat(output, dim=2)