from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(
            input_size=configs.in_channel, 
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
        x, _ = self.lstm(x)
        return self.projection(x)