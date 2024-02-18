import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from.multi_head_attention import MultiHeadAttention
from .decoder import DecoderLayer
from .encoder import EncoderLayer


class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, model_dim, nheads, nlayers, ff_dim, max_seq_len, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.model_dim = model_dim
        self.max_seq_len = max_seq_len

        self.encoder_embedding = nn.Embedding(src_vocab_size, model_dim)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, model_dim)

        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, nheads, ff_dim, dropout) for _ in range(nlayers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(model_dim, nheads, ff_dim, dropout) for _ in range(nlayers)])

        self.fc = nn.Linear(model_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode_positions(self, x):

        pos_encoding = torch.zeros(self.max_seq_len, self.model_dim)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        denom = torch.exp(torch.arange(0, self.model_dim, 2).float() * -math.log(1000.0 / self.model_dim))

        pos_encoding[:, 0::2] = torch.sin(position * denom)
        pos_encoding[:, 1::2] = torch.cos(position * denom)

        self.register_buffer('pos_encoding', pos_encoding.unsqueeze(0))

        return x + pos_encoding[:, :x.size(1)]

    def generate_mask(self, src, tgt):

        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (N, 1, 1, src_len)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask  # (N, 1, 1, tgt_len)

        return src_mask, tgt_mask

    def forward(self, src, tgt):

        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.encode_positions(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.encode_positions(self.decoder_embedding(tgt)))

        encoder_output = src_embedded
        for enc_layer in self.encoder_layers:
            encoder_output = enc_layer(encoder_output, src_mask)

        decoder_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            decoder_output = dec_layer(decoder_output, encoder_output, src_mask, tgt_mask)

        out = self.fc(decoder_output)

        return out


