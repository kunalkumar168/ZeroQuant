import torch
import torch.nn as nn

QUANTIZE_DTYPE = {
    8: torch.int8,
    # 4: torch.uint4,
}

class SymQuantizer:

    def __init__(self, bits, group_num=1):
        self.bits = bits
        self.group_num = group_num

    def quantize(self, input):

        range = (1 << self.bits)
        min_q_value, max_q_value = -range // 2, range // 2 - 1


        input = input.reshape(self.group_num, -1)
        max_input = input.abs().amax(dim=-1, keepdim=True)

        scale = (2 * max_input) / range

        # Converting values to quantized range
        quant_weights = (input / scale).round()

        # Clipping values to the quantized range
        quant_weights = torch.clamp(quant_weights, min_q_value, max_q_value).to(QUANTIZE_DTYPE[self.bits])
        return nn.Parameter(quant_weights, requires_grad=False), scale

    def get_compact_params(self, quantized_weight, quant_scale, return_param=True):
        shape_weight = quantized_weight.shape
        shape_scale = quant_scale.shape

        quantized_weight = torch.flatten(quantized_weight)
        quant_scale = torch.flatten(quant_scale)

        def deconcat_tensors(shape_weight, shape_scale):

            def fn(compact_tensor):
                weight = torch.narrow(compact_tensor, 0, 0, shape_weight.numel()).view(shape_weight)
                scale = torch.narrow(compact_tensor, 0, shape_weight.numel(), shape_scale.numel()).view(shape_scale)
    
                return weight, scale

            return fn
        
        compact_tensor = torch.concat([quantized_weight, quant_scale])
        if return_param:
            compact_tensor = nn.Parameter(compact_tensor, requires_grad=False)
        compact_tensor.deconcat = deconcat_tensors(shape_weight, shape_scale)

        return compact_tensor

class SymDequantizer:
    def __init__(self, dtype, pre_quant_shape):
        self.dtype = dtype
        self.pre_quant_shape = pre_quant_shape

    def dequantize(self, x, scale):
        x = x.to(self.dtype) * scale
        x = x.reshape(self.pre_quant_shape).contiguous()

        return x

        