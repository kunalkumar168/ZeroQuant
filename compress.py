import torch
import torch.nn as nn
import re
from layers import QuantLinear

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
    return isinstance(module, nn.Linear)

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
            group_compress_params = shared_parameters
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
            new_module = QuantLinear(old_module.in_features, old_module.out_features, bias=bias_required)
            new_module.weight.data = old_module.weight.data
            if bias_required:
                new_module.bias.data = old_module.bias.data

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


