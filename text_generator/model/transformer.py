import os
import math

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from .positional_encoder import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(self, ntokens, model_dim, nheads, ff_dim, nlayers, dropout=0.1):
        super().__init__()

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(model_dim, dropout)

        encoder_layers = TransformerEncoderLayer(model_dim, nheads, ff_dim, dropout)
        decoder_layers = TransformerDecoderLayer(model_dim, nheads, ff_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.embedding = nn.Embedding(ntokens, model_dim)
        self.model_dim = model_dim
        self.fc = nn.Linear(model_dim, ntokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        """
        :param src: Tensor, shape [seq_len, batch_size]
        :param src_mask: Tensor, shape [seq_len, seq_len]
        :return: output tensor of shape [seq_len, batch_size, ntokens]
        """
        src = self.embedding(src) * math.sqrt(self.model_dim)
        src = self.pos_encoder(src)
        src_mask = nn.Transformer.generate_square_subsequent_mask(len(src))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(tgt))
        output = self.transformer_encoder(src, src_mask)
        encoder_output = self.fc(output)

        output = self.transformer_decoder(tgt, encoder_output, tgt_mask)
        output = self.fc(output)

        return output
