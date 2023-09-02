# copy from https://github.com/bshall/hubert/
# and https://github.com/mnuhurr/d2v-audio/blob/main/models/transformer.py
import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import DropPath
    

class PatchEmbedding(nn.Module):
    """ 1D Time series to Patch Embedding
    """
    def __init__(self, patch_size=10, in_chans=1, embed_dim=384, norm_layer=None, flatten=True):
        super().__init__()
        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=(patch_size, 1), stride=(patch_size, 1), bias=False
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2) # (B, C, N)
        x = x.transpose(-2, -1)  # BCN -> BNC and normlize on the feature dimention
        x = self.norm(x)
        return x
    

class FeatureExtractor(nn.Module):
    """ 1D Time series to Patch Embedding
    conv_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    """
    def __init__(self, in_chans=1,
            conv_layers=[(128,10,5)] + [(128,3,2)] * 2 + [(128,2,2)] * 1,
            dropout=0., conv_bias=False,
        ):
        super().__init__()
        self.conv_layers = conv_layers
        layers = []
        in_c = in_chans
        for i, (out_c, ks, st) in enumerate(conv_layers):
            if i == 0:
                norm = nn.GroupNorm(out_c, out_c)
            else:
                norm = nn.Identity()
            layers += [self._make_block(in_c, out_c, ks, st, dropout, norm, conv_bias)]
            in_c = out_c
        self.proj = nn.Sequential(*layers)

    def _make_block(self, n_in, n_out, ks, st, dropout=0., norm_layer=None, conv_bias=False):
        block = [
            nn.Conv2d(
                n_in, n_out, kernel_size=(ks,1), stride=(st,1), padding=((ks-1)//2,0), bias=conv_bias
            ),
            nn.Dropout(dropout),
        ]
        if norm_layer and isinstance(norm_layer, nn.LayerNorm):
            block.append(
                nn.Sequential(
                    TransposeLast(),
                    nn.LayerNorm(n_out, elementwise_affine=True),
                    TransposeLast(),
                )
            )
        else:
            block.append(norm_layer)
        block.append(nn.GELU())
        return nn.Sequential(*block)

    def forward(self, x, input_mask=None):
        x = self.proj(x)
        x = x.flatten(2).transpose(-2, -1)  # BCN -> BNC and normlize on the feature dimention
        if input_mask is not None:
            input_mask = self.contract_mask(input_mask, self.conv_layers)
            return x, input_mask
        else:
            return x
    
    def contract_mask(self, mask, conv_layers):
        """
        shrink mask according to the given list of convolution network parameters. assume masked out positions are 
        marked with -inf, and positions containing actual information are marked with 0.
        :param mask: original mask
        :param kernel_sizes: list of kernel sizes in the token encoder
        :param strides: list of strides in the token encoder
        :param paddings: list of paddings in the token encoder
        :return: shrinked mask
        """
        for _, ks, st in conv_layers:
            padding = (ks - 1) // 2
            mask = torch.nn.functional.max_pool1d(mask, kernel_size=ks, stride=st, padding=padding)
        return mask
    

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)
    

class FeatureProjection(nn.Module):
    def __init__(self, in_features=512, out_features=768, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x
    

class ConvPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim=768, kernel_size=128, groups=16):
        super().__init__()
        self.conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size = kernel_size,
            padding = kernel_size // 2,
            groups = groups,
        )
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2))
        x = F.gelu(x[:, :, :-1])
        return x.transpose(1, 2)
    

class SinCosPositionalEmbedding(nn.Module):
    """
    d_model: output dimension for each position
    max_len: maximum position index
    """
    def __init__(self, d_model, max_len, cls_token=False):
        super().__init__()
        assert d_model % 2 == 0
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if cls_token:
            pe = torch.cat([torch.zeros([1, d_model]), pe], axis=0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assume x.shape is (batch, t, d_model)
        return x + self.pe[:x.size(1)]
    

class SinCosPositionalEmbedding2d(nn.Module):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    def __init__(self, d_model, seq_len, cls_token=False):
        super().__init__()

        height = width = int(math.sqrt(seq_len))
        assert(height * width == seq_len), "seq_len should be a square number"
    
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(height, width, d_model)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[:, :,  0:d_model:2] = torch.sin(pos_w * div_term).unsqueeze(0).repeat(height, 1, 1)
        pe[:, :,  1:d_model:2] = torch.cos(pos_w * div_term).unsqueeze(0).repeat(height, 1, 1)
        pe[:, :,   d_model::2] = torch.sin(pos_h * div_term).unsqueeze(1).repeat(1, width, 1)
        pe[:, :, d_model+1::2] = torch.cos(pos_h * div_term).unsqueeze(1).repeat(1, width, 1)
        pe = pe.reshape(-1, d_model*2) # (height*width, d_model*2)

        if cls_token:
            pe = torch.cat([torch.zeros([1, d_model*2]), pe], axis=0)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_ln = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ff_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # 1. self attention
        x = self.attn_ln(x)
        attn_out, attn_weights = self.attn(x, x, x, key_padding_mask=mask)
        x = x + attn_out
        # 2. ff network
        x = x + self.ff(self.ff_ln(x))
        return x, attn_weights.detach()
    
    
class TransformerEncoderLayerTimm(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 drop_path: float = 0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # 1. self attention
        x = self.norm1(x)
        attn_out, attn_weights = self.attn(x, x, x, key_padding_mask=mask)
        x = x + self.drop_path1(attn_out)
        # 2. ff network
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x, attn_weights.detach()
    

class TransformerEncoder(nn.Module):
    def __init__(
        self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        layer_outputs = []
        attn_weights = []
        for layer in self.layers:
            x, w = layer(x, mask=mask)
            layer_outputs.append(x)
            attn_weights.append(w)

        return layer_outputs, attn_weights
    

class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, mask: [torch.Tensor] = None):
        super().__init__()

        self.attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_ln = torch.nn.LayerNorm(d_model)

        self.cross_attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn_ln = torch.nn.LayerNorm(d_model)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(d_ff, d_model)
        )
        self.ff_ln = torch.nn.LayerNorm(d_model)

        self.register_buffer('mask', mask)

    def forward(self,
                x: torch.Tensor,
                xa: torch.Tensor,
                mask: [torch.Tensor] = None,
                xa_mask: [torch.Tensor] = None):

        # 1. self attention
        x = self.attn_ln(x)
        attn_out, self_attn_weights = self.attn(x, x, x, key_padding_mask=mask)
        x = x + attn_out

        # 2. cross attention
        x = self.cross_attn_ln(x)
        attn_out, cross_attn_weights = self.cross_attn(query=x, key=xa, value=xa, key_padding_mask=xa_mask)
        x = x + attn_out

        # 3. ff network
        x = x + self.ff(self.ff_ln(x))

        return x, self_attn_weights.detach(), cross_attn_weights.detach()
    

class TransformerDecoder(torch.nn.Module):
    def __init__(self, d_model: int, n_layers: int, d_ff: int, n_heads: int, max_sequence_length: int, causal: bool = True):
        super().__init__()

        self.positional_encoding = SinCosPositionalEmbedding(d_model, max_sequence_length)
        self.layers = torch.nn.ModuleList([TransformerDecoderLayer(d_model, d_ff, n_heads) for _ in range(n_layers)])

    def forward(self, 
                x: torch.Tensor, 
                xa: torch.Tensor,
                mask: [torch.Tensor] = None, 
                xa_mask: [torch.Tensor] = None):
    
        # x = decoder output, xa = encoder output
        x = self.positional_encoding(x)
        
        # do not collect layers into a list
        attn_weights = []
        x_attn_weights = []
        for layer in self.layers:
            x, w, xw = layer(x, xa, mask=mask, xa_mask=xa_mask)
            attn_weights.append(w)
            x_attn_weights.append(xw)

        return x, attn_weights, x_attn_weights
