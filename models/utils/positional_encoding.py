import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

class PositionalEncoding1d(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        generate 1d positional encoding.
        :param d_model: the number of expected features in the encoder/decoder inputs (required).
        :param dropout: the dropout value (default=0.1).
        :param max_len: the max. length of the incoming sequence (default=5000).

        usage:
        >>> pos_encoder = PositionalEncoding1d(d_model=512, dropout=0.1)
        >>> input = torch.randn(1, 10, 512)
        >>> output = pos_encoder(input)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

class PositionalEncoding2d(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_h=1000, max_w=1000):
        """
        generate 2d positional encoding.
        :param d_model: the number of expected features in the encoder/decoder inputs (required).
        :param dropout: the dropout value (default=0.1).
        :param max_h: the max. height of the incoming sequence (default=5000).
        :param max_w: the max. width of the incoming sequence (default=5000).

        usage:
        >>> pos_encoder = PositionalEncoding2d(d_model=512, dropout=0.1)
        >>> input = torch.randn(1, 10, 512)
        >>> output = pos_encoder(input)
        
        reference:
        https://github.com/kingyiusuen/image-to-latex/blob/main/image_to_latex/models/positional_encoding.py
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        assert d_model % 2 == 0, "d_model must be even"
        d_model_half = d_model // 2
        
        # pe height 
        pe_h = torch.zeros(max_h, d_model_half)
        position_h = torch.arange(0, max_h, dtype=torch.float).unsqueeze(1)
        div_term_h = torch.exp(torch.arange(0, d_model_half, 2).float() * (-math.log(10000.0) / d_model_half))
        pe_h[:, 0::2] = torch.sin(position_h * div_term_h)
        pe_h[:, 1::2] = torch.cos(position_h * div_term_h)
        pe_h = pe_h.unsqueeze(1).permute(2, 0, 1).expand(-1, -1, max_w) # shape: [d_model, max_h, max_w]

        # pe width
        pe_w = torch.zeros(max_w, d_model_half)
        position_w = torch.arange(0, max_w, dtype=torch.float).unsqueeze(1)
        div_term_w = torch.exp(torch.arange(0, d_model_half, 2).float() * (-math.log(10000.0) / d_model_half))
        pe_w[:, 0::2] = torch.sin(position_w * div_term_w)
        pe_w[:, 1::2] = torch.cos(position_w * div_term_w)
        pe_w = pe_w.unsqueeze(1).permute(2, 1, 0).expand(-1, max_h, -1) # shape: [d_model, max_h, max_w]
        pe = torch.cat([pe_h, pe_w], dim=0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(2), : x.size(3)]
        return self.dropout(x)