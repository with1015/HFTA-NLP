# Copyright (c) 2017 Elad Hoffer
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from hfta.ops import get_hfta_op_for

class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (https://arxiv.org/abs/1409.0473)
    Implementation is very similar to tf.contrib.seq2seq.BahdanauAttention
    """
    def __init__(self, query_size, key_size, num_units, normalize=False,
                 batch_first=False, init_weight=0.1, B=1):
        """
        Constructor for the BahdanauAttention.

        :param query_size: feature dimension for query
        :param key_size: feature dimension for keys
        :param num_units: internal feature dimension
        :param normalize: whether to normalize energy term
        :param batch_first: if True batch size is the 1st dimension, if False
            the sequence is first and batch size is second
        :param init_weight: range for uniform initializer used to initialize
            Linear key and query transform layers and linear_att vector
        """
        super(BahdanauAttention, self).__init__()

        self.B = B
        self.normalize = normalize
        self.batch_first = batch_first
        self.num_units = num_units

        self.linear_q = get_hfta_op_for(nn.Linear, B)(query_size, num_units, bias=False)
        self.linear_k = get_hfta_op_for(nn.Linear, B)(key_size, num_units, bias=False)
        nn.init.uniform_(self.linear_q.weight.data, -init_weight, init_weight)
        nn.init.uniform_(self.linear_k.weight.data, -init_weight, init_weight)

        self.linear_att = Parameter(torch.Tensor(num_units))

        self.mask = None

        if self.normalize:
            self.normalize_scalar = Parameter(torch.Tensor(1))
            self.normalize_bias = Parameter(torch.Tensor(num_units))
        else:
            self.register_parameter('normalize_scalar', None)
            self.register_parameter('normalize_bias', None)

        self.reset_parameters(init_weight)

    def reset_parameters(self, init_weight):
        """
        Sets initial random values for trainable parameters.
        """
        stdv = 1. / math.sqrt(self.num_units)
        self.linear_att.data.uniform_(-init_weight, init_weight)

        if self.normalize:
            self.normalize_scalar.data.fill_(stdv)
            self.normalize_bias.data.zero_()

    def set_mask(self, context_len, context):
        """
        sets self.mask which is applied before softmax
        ones for inactive context fields, zeros for active context fields

        :param context_len: b
        :param context: if batch_first: (B x b x t_k x n) else: (B x t_k x b x n)

        self.mask: (B x b x t_k)
        """

        if self.batch_first:
            max_len = context.size(2)
        else:
            max_len = context.size(1)

        #context_len = context_len.repeat(self.B, 1)
        indices = torch.arange(0, max_len, dtype=torch.int64,
                               device=context.device).repeat(self.B, 1)

        self.mask = indices >= (context_len.unsqueeze(1).unsqueeze(2))
        self.mask = self.mask.transpose(0, 1)


    def calc_score(self, att_query, att_keys):
        """
        Calculate Bahdanau score

        :param att_query: fusion_size x b x t_q x n
        :param att_keys: fusion_size x b x t_k x n

        returns: b x t_q x t_k scores
        """

        fusion_size, b, t_k, n = att_keys.size()

        t_q = att_query.size(2)

        att_query = att_query.unsqueeze(3).expand(fusion_size, b, t_q, t_k, n)
        att_keys = att_keys.unsqueeze(2).expand(fusion_size, b, t_q, t_k, n)

        sum_qk = att_query + att_keys

        if self.normalize:
            sum_qk = sum_qk + self.normalize_bias
            linear_att = self.linear_att / self.linear_att.norm()
            linear_att = linear_att * self.normalize_scalar
        else:
            linear_att = self.linear_att

        out = torch.tanh(sum_qk).matmul(linear_att)
        return out

    def forward(self, query, keys):
        """

        :param query: if batch_first: (b x t_q x n) else: (t_q x b x n)
        :param keys: if batch_first: (b x t_k x n) else (t_k x b x n)

        :returns: (context, scores_normalized)
        context: if batch_first: (B x b x t_q x n) else (B x t_q x b x n)
        scores_normalized: if batch_first (B x b x t_q x t_k) else (B x t_q x b x t_k)
        """

        # first dim of keys and query has to be 'batch', it's needed for bmm
        query = torch.stack(query)

        if not self.batch_first:
            keys = keys.transpose(1, 2)
            if query.dim() == 4:
                query = query.transpose(1, 2)

        if query.dim() == 3:
            single_query = True
            query = query.unsqueeze(1)
        else:
            single_query = False

        b = query.size(1)
        t_k = keys.size(2)
        t_q = query.size(2)

        # FC layers to transform query and key
        processed_query = self.linear_q(query.contiguous())
        processed_key = self.linear_k(keys.contiguous())

        # scores: (fusion_size x b x t_q x t_k)
        scores = self.calc_score(processed_query, processed_key)

        if self.mask is not None:
            mask = self.mask.unsqueeze(2).expand(self.B, b, t_q, t_k)
            # I can't use -INF because of overflow check in pytorch
            scores.masked_fill_(mask, -65504.0)

        # Normalize the scores, softmax over t_k
        scores_normalized = F.softmax(scores, dim=-1)

        # Calculate the weighted average of the attention inputs according to
        # the scores
        # context: (B x b x t_q x n)

        context = []
        for idx in range(self.B):
            context.append(torch.bmm(scores_normalized[idx], keys[idx]))
        #context = torch.bmm(scores_normalized, keys)
        context = torch.stack(context)

        if single_query:
            context = context.squeeze(2)
            scores_normalized = scores_normalized.squeeze(2)
        elif not self.batch_first:
            context = context.transpose(1, 2)
            scores_normalized = scores_normalized.transpose(1, 2)

        return context, scores_normalized
