import torch
import torch.nn as nn
import re
from layers import QuantLinear, QuantEmbedding
from quantized_layers import FixQuantizedLinear, FixQuantizedEmbedding, FixQuantizedConv1D
from transformers.pytorch_utils import Conv1D


def recursive_get_module(model, module_name):
    split_module_name = module_name.split('.')
    out = model
    for name in split_module_name:
        out = getattr(out, name)
    return out


def recursive_set_module(model, module_name, new_module):
    split_module_name = module_name.split('.')
    out = model
    for name in split_module_name[:-1]:
        out = getattr(out, name)
    setattr(out, split_module_name[-1], new_module)

def is_valid(module):
    return isinstance(module, nn.Linear) \
              or isinstance(module, nn.Embedding) \
              or isinstance(module,  Conv1D)

def get_module_names(model, module_name):
    module_names = []
    for name, module in model.named_modules():
       is_valid_module = is_valid(module)
       if is_valid_module and re.search(module_name, name):
           module_names.append(name)

    return module_names

def get_quantization_params(model, quantize_params):
    compress_params = []
    for method, params in quantize_params.items():
        shared_parameters = params.get('shared_parameters', {})
        for group_name, group_params in params.get('different_groups', {}).items():
            module_list = []
            for module in group_params['modules']:

                module_names = get_module_names(model, module)
                for module_name in module_names:
                    module_list.append(module_name)
            if module_list:
                group_compress_params = shared_parameters.copy()
                group_compress_params.update(group_params['params'])

                compress_params.append([
                    module_list,
                    { method: group_compress_params }
                ])
    return compress_params


def compress(model, compress_config):
    layer_wise_compress_methods = get_quantization_params(model, compress_config)
    for module_list, compress_params in layer_wise_compress_methods:
        for module in module_list:
            module_replacement(model, module, compress_params)
    return model

def module_replacement(model, module_name, compress_params):
    old_module = recursive_get_module(model, module_name)
    bias_required = False
    if hasattr(old_module, 'bias') and old_module.bias is not None:
        bias_required = True

    new_module = None
    if isinstance(old_module, nn.Linear) or isinstance(old_module, QuantLinear):
        if isinstance(old_module, QuantLinear):
            new_module = old_module
        else:
            new_module = QuantLinear(old_module.in_features, old_module.out_features, bias=bias_required).to(device=old_module.weight.device, dtype=old_module.weight.dtype)
            new_module.weight.data = old_module.weight.data
            if bias_required:
                new_module.bias.data = old_module.bias.data

    elif isinstance(old_module, nn.Embedding) or isinstance(old_module, QuantEmbedding):
        if isinstance(old_module, QuantEmbedding):
            new_module = old_module
        else:
            new_module = QuantEmbedding(old_module.num_embeddings, old_module.embedding_dim, old_module.padding_idx, old_module.max_norm, old_module.norm_type, \
                                        old_module.scale_grad_by_freq, old_module.sparse).to(device=old_module.weight.device, dtype=old_module.weight.dtype)
            new_module.weight.data = old_module.weight.data

    if compress_params:
        for compression_method, compression_params in compress_params.items():
            if compression_method == 'weight_quantization':
                if compression_params.get('enabled', False):
                    new_module.enable_weight_quantization(
                        compression_params['start_bits'],
                        compression_params['target_bits'],
                        compression_params['quantize_weight_in_forward'],
                        compression_params['quantize_groups'],
                        compression_params['quantization_type']
                    )
            elif compression_method == 'activation_quantization':
                if compression_params.get('enabled', False):
                    new_module.enable_activation_quantization(
                        compression_params['bits'],
                        compression_params['quantization_type'],
                        compression_params['range_calibration']
                    )
    recursive_set_module(model, module_name, new_module)



def fix_compression(model, compress_config):
    layer_wise_compress_methods = get_quantization_params(model, compress_config)
    for module_list, compress_params in layer_wise_compress_methods:
        for module in module_list:
            quantized_module_replacement(model, module, compress_params)
    return model

def quantized_module_replacement(model, module_name, compress_params):
    old_module = recursive_get_module(model, module_name)
    weight_quantization_params = compress_params.get('weight_quantization', {})
    
    new_module = old_module
    if weight_quantization_params.get('enabled', False):
        if isinstance(old_module, nn.Linear) or isinstance(old_module, QuantLinear):
            new_module = FixQuantizedLinear(weight_quantization_params['target_bits'], old_module, weight_quantization_params['quantize_groups'], weight_quantization_params['quantization_type'])  

        elif isinstance(old_module, nn.Embedding) or isinstance(old_module, QuantEmbedding):
            new_module = FixQuantizedEmbedding(weight_quantization_params['target_bits'], old_module, weight_quantization_params['quantize_groups'], weight_quantization_params['quantization_type'])
        elif isinstance(old_module, Conv1D):
            new_module = FixQuantizedConv1D(weight_quantization_params['target_bits'], old_module, weight_quantization_params['quantize_groups'], weight_quantization_params['quantization_type'])

    activation_quantization_params = compress_params.get('activation_quantization', {})
   
    if activation_quantization_params.get('enabled', False):
        new_module.enable_activation_quantization(
            activation_quantization_params['bits'],
            activation_quantization_params['quantization_type'],
            activation_quantization_params['range_calibration']
        )
    recursive_set_module(model, module_name, new_module)


