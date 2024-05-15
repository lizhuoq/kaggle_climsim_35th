from torch import nn


class PositionWiseFFN(nn.Module):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_outputs, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
    

class AddNorm(nn.Module):  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
    

class EncoderBlock(nn.Module):  #@save
    def __init__(self, num_hiddens, ffn_num_hiddens, dropout):
        super().__init__()
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, Y):
        return self.addnorm2(Y, self.ffn(Y))
    

class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.embedding = nn.Linear(configs.in_channel, configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        blks = []
        for _ in range(configs.n_layers):
            blks.append(EncoderBlock(configs.d_model, configs.d_ff, configs.dropout))
        self.blks = nn.Sequential(*blks)
        self.projection = nn.Linear(configs.d_model, configs.out_channel)

    def forward(self, x):
        enc_in = self.dropout(self.embedding(x))
        enc_out = self.blks(enc_in)
        return self.projection(enc_out)