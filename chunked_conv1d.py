#!/usr/bin/env python3

import platform

import torch



class ChunkedConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, max_chunk_size=65536):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.dilation = dilation
        self.out_channels = out_channels

        self.min_input_chunk_size = max(kernel_size, stride, padding, dilation) * 2
        self.min_input_chunk_size = max(self.min_input_chunk_size, int(max_chunk_size * 0.1)) # 10% of max_chunk_size
        self.output_chunk_size_reduction = (self.min_input_chunk_size // stride) + 1
        self.input_chunk_size_reduction = self.output_chunk_size_reduction * stride

        max_input_chunk_size = max_chunk_size - 2*padding

        output_start_cutoff = 0
        input_overlap = 0
        if padding > 0:
            if padding == stride:
                output_start_cutoff = 1
                input_overlap = 0
            elif padding < stride:
                output_start_cutoff = 1
                input_overlap = stride - padding
            elif padding > stride:
                output_start_cutoff = (padding - (padding % stride)) // stride + (1 if padding % stride > 0 else 0)
                input_overlap = 0 if padding % stride == 0 else (stride - (padding % stride))

        K = 1 + dilation * (kernel_size - 1)

        if stride <= K:
            L = max_input_chunk_size + padding
            first_chunk_input_size = (L - ((L - K) % stride)) # + K # usable input size
            first_chunk_output_size = (first_chunk_input_size - K) // stride + 1   # actual output size

            L = max_input_chunk_size - input_overlap
            chunk_input_size = (L - ((L - K) % stride))   # + K
            chunk_output_size = (chunk_input_size - K) // stride + 1

        else:   # stride > K
            L = max_input_chunk_size + padding
            if L % stride >= K:
                first_chunk_input_size = (L - (L % stride)) + K
                first_chunk_output_size = (first_chunk_input_size - K) // stride + 1
            else: # L % stride < K
                first_chunk_input_size = (L - (L % stride) - stride) + K
                first_chunk_output_size = (first_chunk_input_size - K) // stride + 1

            L = max_input_chunk_size - input_overlap
            if L % stride >= K:
                chunk_input_size = (L - (L % stride)) + K
                chunk_output_size = (chunk_input_size - K) // stride + 1
            else: # L % stride < K
                chunk_input_size = (L - (L % stride) - stride) + K
                chunk_output_size = (chunk_input_size - K) // stride + 1

        first_chunk_input_size -= padding

        self.first_chunk_input_size = first_chunk_input_size
        self.first_chunk_output_size = first_chunk_output_size
        self.chunk_input_size = chunk_input_size
        self.chunk_output_size = chunk_output_size
        self.max_input_chunk_size = max_input_chunk_size
        self.K = K
        self.stride = stride
        self.input_overlap = input_overlap
        self.output_start_cutoff = output_start_cutoff

        self.register_parameter('weight', self.conv.weight)
        if bias:
            self.register_parameter('bias', self.conv.bias)
        self.register_load_state_dict_post_hook(self._load_state_dict_post_hook)

    def _load_state_dict_post_hook(self, module, incompatible_keys):
        prefix = self.prefix
        remove_keys = list(filter(lambda key: key in (prefix + 'conv.weight', prefix + 'conv.bias'), incompatible_keys.missing_keys))
        for key in remove_keys:
            incompatible_keys.missing_keys.remove(key)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = {}
        # detect if weight_norm is applied
        if hasattr(self, 'weight_g') and hasattr(self, 'weight_v'):
            destination[prefix + 'weight_g'] = self.weight_g.data
            destination[prefix + 'weight_v'] = self.weight_v.data
        else:
            destination[prefix + 'weight'] = self.conv.weight.data
        if hasattr(self, 'bias'):
            destination[prefix + 'bias'] = self.conv.bias.data
        return destination

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.prefix = prefix        # for use in post hook
        weight_g_restored = False
        weight_v_restored = False
        weight_restored = False
        for key, value in state_dict.items():
            if key.startswith(prefix):
                if key == prefix + 'weight_g':
                    self.weight_g.data.copy_(value)
                    weight_g_restored = True
                elif key == prefix + 'weight_v':
                    self.weight_v.data.copy_(value)
                    weight_v_restored = True
                elif key == prefix + 'weight':
                    self.conv.weight.data.copy_(value)
                    self.weight.data.copy_(value)
                    weight_restored = True
                elif key == prefix + 'bias':
                    self.conv.bias.data.copy_(value)
                    self.bias.data.copy_(value)
        if weight_g_restored and weight_v_restored and not weight_restored:
            weight_g = self.weight_g.data
            weight_v = self.weight_v.data
            weight_v_norm = weight_v.view(weight_v.size(0), -1).norm(dim=1, keepdim=True)
            self.conv.weight.data = (weight_v / weight_v_norm.view(-1, 1, 1)) * weight_g.view(-1, 1, 1)
            self.weight.data = self.conv.weight.data

    def forward(self, x):
        input_length = x.size(2)
        output_chunks = []

        conv = self.conv
        first_chunk_input_size = self.first_chunk_input_size
        first_chunk_output_size = self.first_chunk_output_size
        chunk_input_size = self.chunk_input_size
        chunk_output_size = self.chunk_output_size
        max_input_chunk_size = self.max_input_chunk_size
        K = self.K
        stride = self.stride
        input_overlap = self.input_overlap
        output_start_cutoff = self.output_start_cutoff
        min_input_chunk_size = self.min_input_chunk_size

        start = 0
        while start < input_length:
            if start - input_overlap + max_input_chunk_size >= input_length:
                end = input_length
                next_start = end
                output_size = None
            else:
                if start == 0:
                    next_start = start + first_chunk_input_size - K + stride
                    end = start + first_chunk_input_size
                    output_size = first_chunk_output_size
                else:
                    next_start = start + chunk_input_size - K + stride
                    end = start + chunk_input_size
                    output_size = chunk_output_size
                if input_length - next_start < min_input_chunk_size:
                    next_start -= self.input_chunk_size_reduction
                    end -= self.input_chunk_size_reduction
                    output_size -= self.output_chunk_size_reduction
            chunk = x[:, :, max(0,start-input_overlap):end]
            output_chunk = conv(chunk)
            if start > 0:
                output_chunk = output_chunk[:, :, output_start_cutoff:]
            if end < input_length:
                output_chunk = output_chunk[:, :, :output_size]
            output_chunks.append(output_chunk)
            start = next_start
        final_output = torch.cat(output_chunks, dim=2)
        return final_output



class ChunkedConvTranspose1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                 groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None, max_chunk_size=65536):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=0, output_padding=0,
                                             groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode, device=device, dtype=dtype)
        self.dilation = dilation
        self.stride = stride
        self.out_channels = out_channels
        self.K = K = 1 + dilation * (kernel_size - 1)
        self.overlap = K - stride
        self.padding = padding
        self.output_padding = output_padding
        self.max_input_chunk_size = (max_chunk_size - K) // stride + 1
        self.register_parameter('weight', self.conv.weight)
        if bias:
            self.register_parameter('bias', self.conv.bias)
        self.register_load_state_dict_post_hook(self._load_state_dict_post_hook)

    def _load_state_dict_post_hook(self, module, incompatible_keys):
        prefix = self.prefix
        remove_keys = list(filter(lambda key: key in (prefix + 'conv.weight', prefix + 'conv.bias'), incompatible_keys.missing_keys))
        for key in remove_keys:
            incompatible_keys.missing_keys.remove(key)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = {}
        if hasattr(self, 'weight_g') and hasattr(self, 'weight_v'):
            destination[prefix + 'weight_g'] = self.weight_g.data
            destination[prefix + 'weight_v'] = self.weight_v.data
        else:
            destination[prefix + 'weight'] = self.conv.weight.data
        if hasattr(self, 'bias'):
            destination[prefix + 'bias'] = self.conv.bias.data
        return destination

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.prefix = prefix        # for use in post hook
        weight_g_restored = False
        weight_v_restored = False
        weight_restored = False
        for key, value in state_dict.items():
            if key.startswith(prefix):
                if key == prefix + 'weight_g':
                    self.weight_g.data.copy_(value)
                    weight_g_restored = True
                elif key == prefix + 'weight_v':
                    self.weight_v.data.copy_(value)
                    weight_v_restored = True
                elif key == prefix + 'weight':
                    self.conv.weight.data.copy_(value)
                    self.weight.data.copy_(value)
                    weight_restored = True
                elif key == prefix + 'bias':
                    self.conv.bias.data.copy_(value)
                    self.bias.data.copy_(value)
        if weight_g_restored and weight_v_restored and not weight_restored:
            weight_g = self.weight_g.data
            weight_v = self.weight_v.data
            weight_v_norm = weight_v.view(weight_v.size(0), -1).norm(dim=1, keepdim=True)
            self.conv.weight.data = (weight_v / weight_v_norm.view(-1, 1, 1)) * weight_g.view(-1, 1, 1)
            self.weight.data = self.conv.weight.data

    def forward(self, x):
        input_length = x.size(2)
        output_chunks = []

        conv = self.conv
        stride = self.stride
        K = self.K
        overlap = self.overlap
        padding = self.padding
        output_padding = self.output_padding
        input_chunk_size = self.max_input_chunk_size
        final_output_size = (input_length - 1) * stride + K + output_padding
        final_output = torch.zeros([x.size(0), self.out_channels, final_output_size], device=x.device)
        bias = self.bias.view(1, -1, 1) if self.bias is not None else None

        pos = 0
        for start in range(0, input_length, input_chunk_size):
            end = min(input_length, start + input_chunk_size)
            chunk = x[:, :, start:end]
            output_chunk = conv(chunk)
            pos = max(0, pos - overlap)
            end = pos + output_chunk.size(2)
            final_output[:, :, pos:end] += output_chunk
            if pos > 0 and overlap > 0 and bias is not None:
                final_output[:, :, pos:pos + overlap] -= bias
            pos = end
        if padding > 0:
            return final_output[:, :, padding:-padding]
        return final_output



Conv1d = torch.nn.Conv1d
ConvTranspose1d = torch.nn.ConvTranspose1d


if torch.backends.mps.is_available():
    v = lambda version: tuple(map(int, version.split('.')))
    torch_version = v(torch.__version__)
    if hasattr(platform, 'mac_ver'):
        macos_version = v(platform.mac_ver()[0])
        # In case macOS version is below 15.1 or macOS version is at least 15.1, but PyTorch version is 2.5.0 or 2.5.1,
        # replace Conv1d and ConvTranspose1d with chunked approach
        if macos_version < v('15.1.0') or macos_version >= v('15.1.0') and torch_version > v('2.4.1') and torch_version <= v('2.5.1'):
            Conv1d = ChunkedConv1d
            ConvTranspose1d = ChunkedConvTranspose1d



def test():

    def fix_random_seed(seed=24):
        import numpy as np
        import random
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    class Conv1dTestModule(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, cls=torch.nn.Conv1d):
            super().__init__()
            self.conv = cls(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        def forward(self, x):
            return self.conv(x)

    class ConvTranspose1dTestModule(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, cls=torch.nn.ConvTranspose1d):
            super().__init__()
            self.conv = cls(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation)
        def forward(self, x):
            return self.conv(x)

    class Conv1dWeightNormTestModule(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, cls=torch.nn.Conv1d):
            super().__init__()
            self.conv = torch.nn.utils.weight_norm(cls(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation))
        def forward(self, x):
            return self.conv(x)

    class ConvTranspose1dWeightNormTestModule(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, cls=torch.nn.ConvTranspose1d):
            super().__init__()
            self.conv = torch.nn.utils.weight_norm(cls(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation))
        def forward(self, x):
            return self.conv(x)


    def get_module_device(module):
        if next(module.parameters(), None) is not None:
            return next(module.parameters()).device
        elif next(module.buffers(), None) is not None:
            return next(module.buffers()).device
        else:
            raise ValueError("Module has no parameters or buffers.")

    def module_calc_forward(module, input_tensor):
        return module(input_tensor.to(get_module_device(module))).to("cpu")

    def test_state_dict(export_state_dict_module, load_state_dict_module, input_tensor):
        state_dict = export_state_dict_module.state_dict()
        load_state_dict_module.load_state_dict(state_dict)
        output1 = module_calc_forward(export_state_dict_module, input_tensor)
        output2 = module_calc_forward(load_state_dict_module, input_tensor)
        return torch.allclose(output1, output2, atol=1e-4)

    def test_outputs(module1, module2, input_tensor):
        module1.load_state_dict(module2.state_dict())
        output1 = module_calc_forward(module1, input_tensor)
        output2 = module_calc_forward(module2, input_tensor)
        return torch.allclose(output1, output2, atol=1e-4)


    fix_random_seed(20) # for deterministic results

    max_chunk_size = 65536

    params = dict(in_channels=1, out_channels=1, kernel_size=9, stride=3, padding=5, dilation=1)
    input_length = int(max_chunk_size * 2.5)
    x = torch.ones([1, 1, input_length])
    # test ChunkedConv1d load/save state dict against torch.nn.Conv1d
    assert test_state_dict(torch.nn.Conv1d(**params), ChunkedConv1d(**params), x), "ChunkedConv1d load state dict failed"
    assert test_state_dict(ChunkedConv1d(**params), torch.nn.Conv1d(**params), x), "ChunkedConv1d export state dict failed"
    # test ChunkedConv1d as submodule load/save state dict against torch.nn.Conv1d as submodule
    assert test_state_dict(Conv1dTestModule(**params, cls=torch.nn.Conv1d), Conv1dTestModule(**params, cls=ChunkedConv1d), x), "ChunkedConv1d (as submodule) load state dict failed"
    assert test_state_dict(Conv1dTestModule(**params, cls=ChunkedConv1d), Conv1dTestModule(**params, cls=torch.nn.Conv1d), x), "ChunkedConv1d (as submodule) export state dict failed"
    # test weight_norm(ChunkedConv1d) as submodule load/save state dict against weight_norm(torch.nn.Conv1d) as submodule
    assert test_state_dict(Conv1dWeightNormTestModule(**params, cls=torch.nn.Conv1d),
                           Conv1dWeightNormTestModule(**params, cls=ChunkedConv1d), x), "weight_norm(ChunkedConv1d) (as submodule) load state dict failed"
    assert test_state_dict(Conv1dWeightNormTestModule(**params, cls=ChunkedConv1d),
                           Conv1dWeightNormTestModule(**params, cls=torch.nn.Conv1d), x), "weight_norm(ChunkedConv1d) (as submodule) export state dict failed"
    assert test_outputs(Conv1dWeightNormTestModule(**params, cls=torch.nn.Conv1d),
                        Conv1dWeightNormTestModule(**params, cls=ChunkedConv1d), x), "weight_norm(ChunkedConv1d) (as submodule) output does not match"
    assert test_outputs(torch.nn.Conv1d(**params), ChunkedConv1d(**params, max_chunk_size=max_chunk_size), x), "ChunkedConv1d output does not match"
    for delta in range(-100, 100):
        x = torch.ones([1, 1, max_chunk_size + delta])
        assert test_outputs(torch.nn.Conv1d(**params), ChunkedConv1d(**params, max_chunk_size=max_chunk_size), x), "ChunkedConv1d output does not match"

    # test ChunkedConvTranspose1d load/save state dict against torch.nn.ConvTranspose1d
    # NOTE: output padding must be smaller than either stride or dilation
    params = dict(in_channels=2, out_channels=3, kernel_size=4, stride=2, padding=3, output_padding=0, dilation=1)
    input_length = int(max_chunk_size * 2.5)
    x = torch.ones([1, 2, input_length])
    # test ChunkedConvTranspose1d load/save state dict against torch.nn.ConvTranspose1d
    assert test_state_dict(torch.nn.ConvTranspose1d(**params), ChunkedConvTranspose1d(**params), x), "ChunkedConvTranspose1d load state dict failed"
    assert test_state_dict(ChunkedConvTranspose1d(**params), torch.nn.ConvTranspose1d(**params), x), "ChunkedConvTranspose1d export state dict failed"
    # test ChunkedConvTranspose1d as submodule load/save state dict against torch.nn.ConvTranspose1d as submodule
    assert test_state_dict(ConvTranspose1dTestModule(**params, cls=torch.nn.ConvTranspose1d),
                           ConvTranspose1dTestModule(**params, cls=ChunkedConvTranspose1d), x), "ChunkedConvTranspose1d (as submodule) load state dict failed"
    assert test_state_dict(ConvTranspose1dTestModule(**params, cls=ChunkedConvTranspose1d),
                           ConvTranspose1dTestModule(**params, cls=torch.nn.ConvTranspose1d), x), "ChunkedConvTranspose1d (as submodule) export state dict failed"
    # test weight_norm(ChunkedConvTranspose1d) as submodule load/save state dict against weight_norm(torch.nn.ConvTranspose1d) as submodule
    assert test_state_dict(ConvTranspose1dWeightNormTestModule(**params, cls=torch.nn.ConvTranspose1d),
                           ConvTranspose1dWeightNormTestModule(**params, cls=ChunkedConvTranspose1d), x), "weight_norm(ChunkedConvTranspose1d) (as submodule) load state dict failed"
    assert test_state_dict(ConvTranspose1dWeightNormTestModule(**params, cls=ChunkedConvTranspose1d),
                           ConvTranspose1dWeightNormTestModule(**params, cls=torch.nn.ConvTranspose1d), x), "weight_norm(ChunkedConvTranspose1d) (as submodule) export state dict failed"
    assert test_outputs(ConvTranspose1dWeightNormTestModule(**params, cls=torch.nn.ConvTranspose1d),
                        ConvTranspose1dWeightNormTestModule(**params, cls=ChunkedConvTranspose1d), x), "weight_norm(ChunkedConvTranspose1d) (as submodule) output does not match"
    assert test_outputs(torch.nn.ConvTranspose1d(**params), ChunkedConvTranspose1d(**params, max_chunk_size=max_chunk_size), x), "ChunkedConvTranspose1d output does not match"
    for delta in range(-100, 100):
        x = torch.ones([1, 2, max_chunk_size + delta])
        assert test_outputs(torch.nn.ConvTranspose1d(**params), ChunkedConvTranspose1d(**params, max_chunk_size=max_chunk_size), x), "ChunkedConvTranspose1d output does not match"



if __name__ == '__main__':
    test()
