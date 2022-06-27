import torch
import math
import numbers
import warnings

from typing import List, Tuple, Optional, overload, Union, cast

from torch import Tensor
from torch.nn import init
from torch.nn.utils.rnn import PackedSequence
from torch.nn.parameter import Parameter
from threading import Thread

_rnn_impls = {
    'RNN_TANH': torch._VF.rnn_tanh,
    'RNN_RELU': torch._VF.rnn_relu,
}

def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)

class RNNBase(torch.nn.Module):

    def __init__ (
            self,
            mode: str,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = True,
            batch_first: bool = False,
            dropout: float = 0,
            bidirectional: bool = False,
            proj_size: int = 0,
            device=None,
            dtype=None,
            B: int = 1) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        num_directions = 2 if bidirectional else 1
        self.B = B
        self.stream_set = [torch.cuda.Stream() for i in range(B)]

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")

        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        if proj_size < 0:
            raise ValueError("proj_size should be a positive integer or zero to disable projections")
        if proj_size >= hidden_size:
            raise ValueError("proj_size has to be smaller than hidden_size")

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'RNN_TANH':
            gate_size = hidden_size
        elif mode == 'RNN_RELU':
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self._flat_weights_names = []
        self._all_weights = []

        for layer in range(num_layers):
            for direction in range(num_directions):
                real_hidden_size = proj_size if proj_size > 0 else hidden_size
                layer_input_size = input_size if layer == 0 else real_hidden_size * num_directions

                w_ih = Parameter(torch.empty((B, gate_size, layer_input_size), **factory_kwargs))
                w_hh = Parameter(torch.empty((B, gate_size, real_hidden_size), **factory_kwargs))
                b_ih = Parameter(torch.empty((B, gate_size), **factory_kwargs))
                # Second bias vector included for CuDNN compatibility. Only one
                # bias vector is needed in standard definition.
                b_hh = Parameter(torch.empty((B, gate_size), **factory_kwargs))
                layer_params: Tuple[Tensor, ...] = ()
                if self.proj_size == 0:
                    if bias:
                        layer_params = (w_ih, w_hh, b_ih, b_hh)
                    else:
                        layer_params = (w_ih, w_hh)
                else:
                    w_hr = Parameter(torch.empty((proj_size, hidden_size), **factory_kwargs))
                    if bias:
                        layer_params = (w_ih, w_hh, b_ih, b_hh, w_hr)
                    else:
                        layer_params = (w_ih, w_hh, w_hr)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                if self.proj_size > 0:
                    param_names += ['weight_hr_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
        self.flatten_parameters()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        expected_input_dim = 4 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
                          msg: str = 'Expected hidden size {}, got {}') -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    # get_expected_hidden_size -> self.B 
    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    # get_expected_hidden_size -> self.B
    # (B, num_layers x num_directions, batch, hidden_size)
    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = batch_sizes
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        if self.proj_size > 0:
            expected_hidden_size = (self.B, self.num_layers * num_directions,
                                    mini_batch, self.proj_size)
        else:
            expected_hidden_size = (self.B, self.num_layers * num_directions,
                                    mini_batch, self.hidden_size)
        return expected_hidden_size

    def __setattr__(self, attr, value):
        if hasattr(self, "_flat_weights_names") and attr in self._flat_weights_names:
            # keep self._flat_weights up to date if you do self.weight = ...
            idx = self._flat_weights_names.index(attr)
            self._flat_weights[idx] = value
        super(RNNBase, self).__setattr__(attr, value)

    def flatten_parameters(self) -> None:
        """Resets parameter data pointer so that they can use faster code paths.
        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        # Short-circuits if _flat_weights is only partially instantiated
        if len(self._flat_weights) != len(self._flat_weights_names):
            return

        for w in self._flat_weights:
            if not isinstance(w, Tensor):
                return
        # Short-circuits if any tensor in self._flat_weights is not acceptable to cuDNN
        # or the tensors in _flat_weights are of different dtypes

        first_fw = self._flat_weights[0]
        dtype = first_fw.dtype
        for fw in self._flat_weights:
            if (not isinstance(fw.data, Tensor) or not (fw.data.dtype == dtype) or
                    not fw.data.is_cuda or
                    not torch.backends.cudnn.is_acceptable(fw.data)):
                return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = set(p.data_ptr() for p in self._flat_weights)
        if len(unique_data_ptrs) != len(self._flat_weights):
            return

        with torch.cuda.device_of(first_fw):
            import torch.backends.cudnn.rnn as rnn

            # Note: no_grad() is necessary since _cudnn_rnn_flatten_weight is
            # an inplace operation on self._flat_weights
            with torch.no_grad():
                if torch._use_cudnn_rnn_flatten_weight():
                    num_weights = 4 if self.bias else 2
                    if self.proj_size > 0:
                        num_weights += 1
                    torch._cudnn_rnn_flatten_weight(
                        self._flat_weights, num_weights,
                        self.input_size, rnn.get_cudnn_mode(self.mode),
                        self.hidden_size, self.proj_size, self.num_layers,
                        self.batch_first, bool(self.bidirectional))

    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]):
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)

    # TODO -> improving GPU util
    # Note: input shape [B, *, batch_size, *]
    def forward(self,
                input: Union[Tensor, PackedSequence],
                hx: Optional[Tensor] = None) -> Tuple[Union[Tensor, PackedSequence], Tensor]:
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0])
        else:
            input = cast(Tensor, input)
            batch_sizes = [input.size(2)]
            max_batch_size = input.size(1) if self.batch_first else input.size(2)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            input = cast(Tensor, input)
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.B, self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        assert hx is not None
        input = cast(Tensor, input)
        self.check_forward_args(input, hx, batch_sizes)
        _impl = _rnn_impls[self.mode]


        # TODO: gpu util is ahan orhm... sibal

        results = []
        #results = {}
        #thread_pool = []

        def run(B_iter, input, hx, flat_weights):
            stream = self.stream_set[B_iter]
            with torch.cuda.stream(stream):
                result = _impl(input, hx, flat_weights, self.bias, self.num_layers,
                               self.dropout, self.training, self.bidirectional, self.batch_first)
                results[B_iter] = result

        #print(input.shape, hx[0].shape, self._flat_weights[0][0].shape)

        for B_iter in range(self.B):
            flat_weights = [weight[B_iter] for weight in self._flat_weights]
            if batch_sizes is None:
                result = _impl(input, hx[B_iter], flat_weights, self.bias, self.num_layers,
                               self.dropout, self.training, self.bidirectional, self.batch_first)
            else:
                result = _impl(input, batch_sizes, hx[B_iter], flat_weights, self.bias,
                               self.num_layers, self.dropout, self.training, self.bidirectional)
            results.append(result)
            #t = Thread(target=run, args=(B_iter, input, hx[B_iter], flat_weights))
            #thread_pool.append(t)

        """
        for t in thread_pool:
            t.start()

        for t in thread_pool:
            t.join()
        """

        output: Union[Tensor, PackedSequence]
        output = [result[0] for result in results]
        hidden = [result[1] for result in results]
        #output = [v[0] for k, v in results.items()]
        #hidden = [v[1] for k, v in results.items()]

        if is_packed:
            output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

class RNN(RNNBase):
    def __init__(self, *args, **kwargs):
        if 'proj_size' in kwargs:
            raise ValueError("proj_size argument is only supported for LSTM, not RNN or GRU")
        self.nonlinearity = kwargs.pop('nonlinearity', 'tanh')
        if self.nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif self.nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(self.nonlinearity))
        super(RNN, self).__init__(mode, *args, **kwargs)


# DEBUG
if __name__ == "__main__":
    import time
    max_B = 100
    for i in range(1, max_B):
        #rnn = RNNBase(mode="RNN_RELU", input_size=10, hidden_size=20, num_layers=2, B=i)
        B = 40
        rnn = RNN(10, 20, 2, B=B).cuda()
        dummy = torch.randn(5, 3, 10).cuda()
        #h0 = torch.randn(3, 2, 3, 20)
        start = time.time()
        output, hn = rnn(dummy)
        end = time.time()
        print("Iter:", i, "| B:", B, "| Time elapsed:", end - start)
