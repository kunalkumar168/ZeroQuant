import torch
import torch.nn as nn

class QuantSym(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, min_val, max_val, group_num=1):
        if bits == 32:
            return x
        
        input_shape = input.shape
        range = (1 << bits)
        min_q_value, max_q_value = -range // 2, range // 2 - 1


        if min_val is not None:
            max_input = torch.max(min_val.abs(), max_val).reshape(-1)
        else:
            input = input.reshape(group_num, -1)
            max_input = input.abs().amax(dim=-1, keepdim=True)

        scale = (2 * max_input) / range

        # Converting values to quantized range
        output = (input / scale).round()

        # Clipping values to the quantized range
        output = torch.clamp(output, min_q_value, max_q_value)

        # Rescaling values to the original range
        output = output * scale

        # Reshaping the output to the original shape
        output = output.reshape(input_shape).contiguous()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None
    
class StaticQuantActivation(nn.Module):
    def __init__(self, range_momentum_param=0.95, quantization_type='symmetric'):
        super(StaticQuantActivation, self).__init__()
        self.range_momentum_param = range_momentum_param
        if quantization_type == 'symmetric':
            self.activation_quantizer = QuantSym.apply

        self.register_buffer('min_max', torch.zeros(2))

    def forward(self, x, bits, *args, **kwargs):
        if self.training:
            min_val = x.data.min()
            max_val = x.data.max()
            self.min_max[0] = self.range_momentum_param * self.min_max[0] + (1 - self.range_momentum_param) * min_val
            self.min_max[1] = self.range_momentum_param * self.min_max[1] + (1 - self.range_momentum_param) * max_val
        out = self.activation_quantizer(x, bits, self.min_max[0], self.min_max[1])
        return out