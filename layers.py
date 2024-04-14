import torch.nn as nn
import torch
from quantizer import QuantSym, StaticQuantActivation


class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(QuantLinear, self).__init__(in_features, out_features, bias)

        self.quantization_enabled = False
        self.activation_enabled = False
        self.quant_bits = None

    def enable_weight_quantization(self, orig_bits, quant_bits, weight_quant_enabled_forward, group_num, quant_type='symmetric'):
        self.weight.orig_bits = orig_bits
        self.weight.quant_bits = quant_bits
        self.weight_quant_enabled_forward = weight_quant_enabled_forward
        if weight_quant_enabled_forward:
            if quant_type == 'symmetric':
                self.weight_quantizer = QuantSym.apply

            self.weight_quant_num_group = group_num

    def fix_weight_quantization(self):
        self.weight.data = self.weight_quantizer(self.weight, self.weight.quant_bits, None, None, self.weight_quant_num_group).data
        self.weight_quant_enabled_forward = False

    
    def enable_activation_quantization(self, quant_bits, quant_type='symmetric', calibration='dynamic'):
        self.activation_quant_bits = quant_bits
        self.activation_quant_method = calibration

        if calibration == 'static':
            self.activation_quantoizer = StaticQuantActivation(quantization_type=quant_type)
        else:
            if quant_type == 'symmetric':
                self.activation_quantizer = QuantSym.apply
            else:
                raise NotImplementedError
            
    def forward(self, x):
        if self.weight_quant_enabled_forward:
            weight = self.weight_quantizer(self.weight, self.weight.quant_bits, None, None, self.weight_quant_num_group)
            bias = self.bias

        if self.activation_enabled:
            num_groups = None
            if self.activation_quant_method == 'dynamic':
                num_groups = x.numel() // x.shape[-1]
            else:
                num_groups = 1
            x = self.activation_quantizer(x, self.activation_quant_bits, None, None, num_groups)

        return nn.functional.linear(x, weight, bias)








