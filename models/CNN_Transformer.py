from torch import nn
import torch


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv1d(expand_c, squeeze_c, 1)
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv1d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x shape: batch_size, d_model, seq_len
        Return:
            shape: batch_size, d_model, seq_len
        """
        scale = nn.functional.adaptive_avg_pool1d(x, 1)
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class ConvBlock(nn.Module):
    def __init__(self, d_model, d_ff, droppath_rate, kernel_size, stride, padding):
        super().__init__()
        self.expand_conv = nn.Sequential(nn.Conv1d(d_model, d_ff, kernel_size=1), nn.SiLU())
        self.dwconv = nn.Sequential(nn.Conv1d(d_ff, d_ff, kernel_size=kernel_size, stride=stride, groups=d_ff, padding=padding), nn.SiLU())
        self.se = SqueezeExcitation(d_model, d_ff)
        self.project_conv = nn.Sequential(nn.Conv1d(d_ff, d_model, kernel_size=1), nn.SiLU())
        if droppath_rate > 0:
            self.droppath = DropPath(droppath_rate)
        else:
            self.droppath = nn.Identity()

    def forward(self, x):
        """
        Input:
            x shape: batch_size, d_model, seq_len
        Return:
            shape: batch_size, d_model, seq_len
        """
        res = self.expand_conv(x)
        res = self.dwconv(res)
        res = self.se(res)
        res = self.project_conv(res)
        return x + self.droppath(res)
    

class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.embedding = nn.Conv1d(configs.in_channel, configs.d_model, kernel_size=1)
        n_layers = 2
        num_conv_blocks = n_layers * 3
        self.conv1 = ConvBlock(configs.d_model, configs.d_ff, configs.drop_path * (1 / num_conv_blocks), kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(configs.d_model, configs.d_ff, configs.drop_path * (2 / num_conv_blocks), kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(configs.d_model, configs.d_ff, configs.drop_path * (3 / num_conv_blocks), kernel_size=3, stride=1, padding=1)
        self.transformerencoder1 = nn.TransformerEncoderLayer(
            d_model=configs.d_model, nhead=configs.nhead, dim_feedforward=configs.d_ff, dropout=configs.dropout, activation=nn.functional.silu, batch_first=True
        )
        self.conv4 = ConvBlock(configs.d_model, configs.d_ff, configs.drop_path * (4 / num_conv_blocks), kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvBlock(configs.d_model, configs.d_ff, configs.drop_path * (5 / num_conv_blocks), kernel_size=3, stride=1, padding=1)
        self.conv6 = ConvBlock(configs.d_model, configs.d_ff, configs.drop_path * (6 / num_conv_blocks), kernel_size=3, stride=1, padding=1)
        self.transformerencoder2 = nn.TransformerEncoderLayer(
            d_model=configs.d_model, nhead=configs.nhead, dim_feedforward=configs.d_ff, dropout=configs.dropout, activation=nn.functional.silu, batch_first=True
        )

        self.projection = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2), 
            nn.SiLU(), 
            nn.Linear(configs.d_model // 2, configs.d_model // 4), 
            nn.SiLU(), 
            nn.Linear(configs.d_model // 4, configs.out_channel)
        )
        

    def forward(self, x: torch.Tensor):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel)
        Return shape:
            (batch_size, seq_len, out_channel)
        """
        x = self.embedding(x.transpose(1, 2))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.transformerencoder1(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.transformerencoder2(x.transpose(1, 2))
        x = self.projection(x)
        return x