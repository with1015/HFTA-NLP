import math
import warnings
import numbers
from typing import List, Tuple, Optional, overload

import torch

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.nn.parameter import Parameter
from .rnn import RNNBase, RNN


class LSTM(RNNBase):

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)

    def get_expected_cell_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = batch_sizes
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.B, self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_forward_args(self,  # type: ignore[override]
                           input: Tensor,
                           hidden: list,
                           batch_sizes: Optional[Tensor],
                           ):
        self.check_input(input, batch_sizes)

        if isinstance(hidden, tuple):
            self.check_hidden_size(hidden[0], self.get_expected_hidden_size(input, batch_sizes),
                                   'Expected hidden[0] size {}, got {}')
            self.check_hidden_size(hidden[1], self.get_expected_cell_size(input, batch_sizes),
                                   'Expected hidden[1] size {}, got {}')

    def permute_hidden(self,  # type: ignore[override]
                       hx: list,
                       permutation: Optional[Tensor]
                       ) -> Tuple[Tensor, Tensor]:
        if permutation is None:
            return hx
        return apply_permutation(hx[0], permutation), apply_permutation(hx[1], permutation)

    def forward(self, input, hx=None):  # noqa: F811
        batch_sizes = None
        max_batch_size = input[0].size(0) if self.batch_first else input[0].size(1)
        sorted_indices = None
        unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
            h_zeros = torch.zeros(self.B, self.num_layers * num_directions,
                                  max_batch_size, real_hidden_size,
                                  dtype=input.dtype, device=input.device)
            c_zeros = torch.zeros(self.B, self.num_layers * num_directions,
                                  max_batch_size, self.hidden_size,
                                  dtype=input.dtype, device=input.device)
            hx = (h_zeros, c_zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, max_batch_size)

        results = []

        for B_iter in range(self.B):
            flat_weights = [weight[B_iter].contiguous() for weight in self._flat_weights]
            hx_b = (hx[0][B_iter], hx[1][B_iter])

            if batch_sizes is None:
                result = torch._VF.lstm(input[B_iter], hx_b, flat_weights, self.bias, self.num_layers,
                                        self.dropout, self.training, self.bidirectional, self.batch_first)
            else:
                result = torch._VF.lstm(input[B_iter], batch_sizes, hx_b, flat_weights, self.bias,
                                        self.num_layers, self.dropout, self.training, self.bidirectional)
            results.append(result)

        output = [result[0] for result in results]
        hidden = [result[1:] for result in results]
        hidden = tuple(map(torch.stack, zip(*hidden)))

        #if isinstance(orig_input, PackedSequence):
        #    output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        #    return output_packed, self.permute_hidden(hidden, unsorted_indices)
        #else:
        return output, self.permute_hidden(hidden, unsorted_indices)


# DEBUG
if __name__ == "__main__":
    import time
    print("Test HFTA LSTM...")
    max_iter = 100

    for i in range(max_iter):
        start = time.time()
        B = 10
        lstm = LSTM(10, 20, 2, B=B).cuda()
        input = torch.randn(5, 3, 10).cuda()
        output, (hn, cn) = lstm(input)
        end = time.time()
        print("Iter:", i, "| B:", B, "| Time elapsed:", end - start, "sec")
