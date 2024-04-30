import torch
import torch.nn as nn
from fix_quantizer import SymQuantizer, SymDequantizer
import torch.nn.functional as F
from quantizer import QuantSym, StaticQuantActivation
from transformers.pytorch_utils import Conv1D

quant_weight_mappings = {}

def get_quant_weight_wrapper(pre_quant_weight, quantizer):
   if id(pre_quant_weight) in quant_weight_mappings:
       return quant_weight_mappings[id(pre_quant_weight)]
   
   else:
        quant_weight_responses = quantizer.quantize(pre_quant_weight)
        quant_weight_mappings[id(pre_quant_weight)] = quant_weight_responses
        return quant_weight_mappings[id(pre_quant_weight)]
   
def get_quantizer(quant_type, bits, group_num):
    if quant_type == 'symmetric':
        return SymQuantizer(bits, group_num)
    else:
        raise NotImplementedError
    
def get_dequantizer(quant_type, dtype, pre_quant_shape, bits):
    if quant_type == 'symmetric':
        return SymDequantizer(dtype, pre_quant_shape, bits)
    else:
        raise NotImplementedError

class FixQuantizedLinear(nn.Linear):
    def __init__(self, bits, pre_quant_layer, group_num=1, quant_type='symmetric'):
        super(FixQuantizedLinear, self).__init__(
            in_features=pre_quant_layer.in_features,
            out_features=pre_quant_layer.out_features,
            bias=pre_quant_layer.bias is not None,
            device=pre_quant_layer.weight.device,
            dtype=pre_quant_layer.weight.dtype
        )
        self.bits = bits
        self.group_num = group_num
        self.quantizer = get_quantizer(quant_type, bits, group_num)
        self.weight, self.scale = get_quant_weight_wrapper(pre_quant_layer.weight, self.quantizer)
        self.bias = pre_quant_layer.bias
        self.weight.dequantizer = get_dequantizer(quant_type, pre_quant_layer.weight.dtype, pre_quant_layer.weight.shape, bits)
        self.activation_quant_enabled = False

    def enable_activation_quantization(self, quant_bits, quant_type='symmetric', calibration='dynamic'):
        self.activation_quant_bits = quant_bits
        self.activation_quant_method = calibration
        self.activation_quant_enabled = True

        if calibration == 'static':
            self.activation_quantizer = StaticQuantActivation(quantization_type=quant_type)
        else:
            if quant_type == 'symmetric':
                self.activation_quantizer = QuantSym.apply
            else:
                raise NotImplementedError

    def forward(self, x):
        temp_dequantized_weight = self.weight.dequantizer.dequantize(self.weight, self.scale)
        if self.activation_quant_enabled:
            num_groups = None
            if self.activation_quant_method == 'dynamic':
                num_groups = x.numel() // x.shape[-1]
            else:
                num_groups = 1
            x = self.activation_quantizer(x, self.activation_quant_bits, None, None, num_groups)

        return nn.functional.linear(x, temp_dequantized_weight, self.bias)
    

class FixQuantizedEmbedding(nn.Embedding):
    def __init__(self, bits, pre_quant_layer, group_num=1, quant_type='symmetric'):
        super(FixQuantizedEmbedding, self).__init__(
            num_embeddings=pre_quant_layer.num_embeddings,
            embedding_dim=pre_quant_layer.embedding_dim,
            padding_idx=pre_quant_layer.padding_idx,
            max_norm=pre_quant_layer.max_norm,
            norm_type=pre_quant_layer.norm_type,
            scale_grad_by_freq=pre_quant_layer.scale_grad_by_freq,
            sparse=pre_quant_layer.sparse,
            _weight=pre_quant_layer.weight,
            device=pre_quant_layer.weight.device,
            dtype=pre_quant_layer.weight.dtype,
            _freeze=True
        )
        self.bits = bits
        self.group_num = group_num
        self.quantizer = get_quantizer(quant_type, bits, group_num)
        self.weight, self.scale = get_quant_weight_wrapper(pre_quant_layer.weight, self.quantizer)

        self.weight.dequantizer = get_dequantizer(quant_type, pre_quant_layer.weight.dtype, pre_quant_layer.weight.shape, bits)

    def forward(self, x):
        temp_dequantized_weight = self.weight.dequantizer.dequantize(self.weight, self.scale)

        return F.embedding(x, temp_dequantized_weight, self.padding_idx, self.max_norm, self.norm_type,
                           self.scale_grad_by_freq, self.sparse)
    

class FixQuantizedConv1D(Conv1D):
    def __init__(self, bits, pre_quant_layer, group_num=1, quant_type='symmetric'):
        super(FixQuantizedConv1D, self).__init__(
            nf=pre_quant_layer.nf,
            nx=pre_quant_layer.weight.shape[0]
        )
        self.bits = bits
        self.group_num = group_num
        self.quantizer = get_quantizer(quant_type, bits, group_num)
        self.weight, self.scale = get_quant_weight_wrapper(pre_quant_layer.weight, self.quantizer)
        self.bias = pre_quant_layer.bias
        self.weight.dequantizer = get_dequantizer(quant_type, pre_quant_layer.weight.dtype, pre_quant_layer.weight.shape, bits)
        self.activation_quant_enabled = False

    def enable_activation_quantization(self, quant_bits, quant_type='symmetric', calibration='dynamic'):
        self.activation_quant_bits = quant_bits
        self.activation_quant_method = calibration
        self.activation_quant_enabled = True

        if calibration == 'static':
            self.activation_quantizer = StaticQuantActivation(quantization_type=quant_type)
        else:
            if quant_type == 'symmetric':
                self.activation_quantizer = QuantSym.apply
            else:
                raise NotImplementedError

    def forward(self, x):
        temp_dequantized_weight = self.weight.dequantizer.dequantize(self.weight, self.scale)
        if self.activation_quant_enabled:
            num_groups = None
            if self.activation_quant_method == 'dynamic':
                num_groups = x.numel() // x.shape[-1]
            else:
                num_groups = 1
            x = self.activation_quantizer(x, self.activation_quant_bits, None, None, num_groups)

        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), temp_dequantized_weight)
        x = x.view(size_out)
        return x