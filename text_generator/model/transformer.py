import os
import math

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from .positional_encoder import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(self, ntokens, model_dim, nheads, ff_dim, nlayers, seq_len, dropout=0.1):
        super().__init__()

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(model_dim, seq_len, dropout)

        encoder_layers = TransformerEncoderLayer(model_dim, nheads, ff_dim, dropout)
        decoder_layers = TransformerDecoderLayer(model_dim, nheads, ff_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.enc_embedding = nn.Embedding(ntokens, model_dim)
        self.dec_embedding = nn.Embedding(ntokens, model_dim)
        self.model_dim = model_dim
        self.fc = nn.Linear(model_dim, ntokens)

    def forward(self, src, tgt):
        """
        :param src: Tensor, shape [seq_len, batch_size]
        :param tgt: Tensor, shape [seq_len, batch_size]
        :return: output tensor of shape [seq_len, batch_size, ntokens]
        """
        src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(0))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0))

        src = self.enc_embedding(src) * math.sqrt(self.model_dim)
        src = self.pos_encoder(src)
        tgt = self.dec_embedding(tgt) * math.sqrt(self.model_dim)
        tgt = self.pos_encoder(tgt)

        encoder_output = self.transformer_encoder(src, src_mask)
        decoder_output = self.transformer_decoder(tgt, encoder_output, tgt_mask=tgt_mask, memory_mask=src_mask)
        output = self.fc(decoder_output)

        return output
