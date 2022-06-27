import torch
import torch.nn as nn
import random
import time

from torch.nn.utils.rnn import pack_padded_sequence
from hfta.ops import get_hfta_op_for

class Encoder(nn.Module):
    def __init__(self,
                input_size: int,
                emb_size: int,
                hidden_size: int,
                n_layers: int,
                dropout: float,
                batch_first: bool=False,
                B: int = 1):
        super().__init__()

        self.hidden_dim = hidden_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.B = B

        self.embedding = get_hfta_op_for(nn.Embedding, B)(input_size, emb_size)
        self.lstm = get_hfta_op_for(nn.LSTM, B)(emb_size, hidden_size, n_layers, dropout=dropout)
        self.dropout = get_hfta_op_for(nn.Dropout, B)(p=dropout)

    def forward(self, input):
        batch_size = input.size(1)
        stacked = [input for i in range(self.B)]
        input = torch.stack(stacked)
        embedded = self.dropout(self.embedding(input))

        start = time.time()
        output, hx = self.lstm(embedded)

        return hx


class Decoder(nn.Module):
    def __init__(self,
                output_size: int,
                emb_size: int,
                hidden_size: int,
                n_layers: int,
                dropout: float,
                B: int = 1):
        super().__init__()

        self.output_dim = output_size
        self.hid_dim = hidden_size
        self.n_layers = n_layers
        self.B = B

        self.embedding = get_hfta_op_for(nn.Embedding, B)(output_size, emb_size)
        self.lstm = get_hfta_op_for(nn.LSTM, B)(emb_size, hidden_size, n_layers, dropout=dropout)
        self.fc = get_hfta_op_for(nn.Linear, B)(hidden_size, output_size)
        self.dropout = get_hfta_op_for(nn.Dropout, B)(p=dropout)

    def forward(self, input, hx):
        input = input.unsqueeze(0)
        stacked = [input for i in range(self.B)]
        input = torch.stack(stacked)
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.lstm(embedded, hx)
        output = torch.stack((output)).squeeze(1)
        prediction = self.fc(output)
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 n_layers: int,
                 enc_emb_dim: int,
                 dec_emb_dim: int,
                 enc_dropout: float = 0.5,
                 dec_dropout: float = 0.5,
                 B: int = 1):
        super().__init__()

        self.B = B
        self.encoder = Encoder(input_size, enc_emb_dim, hidden_size, n_layers, enc_dropout, B=B)
        self.decoder = Decoder(output_size, dec_emb_dim, hidden_size, n_layers, dec_dropout, B=B)

    def forward(self, src, tgt, ratio=0.5):
        batch_size = tgt.shape[1]
        tgt_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(self.B, tgt_len, batch_size, tgt_vocab_size).cuda()

        hx = self.encoder(src)
        input = tgt[0, :]

        for t in range(1, tgt_len):
            start = time.time()
            decoder_outputs, (hidden, cell) = self.decoder(input, hx)
            for idx in range(decoder_outputs.size(0)):
                outputs[idx][t] = decoder_outputs[idx]
                teacher_force = random.random() < ratio
                top1 = outputs.argmax(2)
                #input = tgt[t] if teacher_force else top1

        return outputs
