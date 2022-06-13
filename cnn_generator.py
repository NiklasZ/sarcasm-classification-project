from typing import Tuple, List
import torch.nn as nn


# input_shape should be [C=channels, H=height, W=width]
def create_cnn_model(model_params: dict, input_shape: Tuple[int, int, int], classes: int) -> nn.Sequential:
    layers = nn.ModuleList()

    # Convolution Section
    current_shape = input_shape
    for i, m in enumerate(model_params['modules']):
        if m['type'] == 'conv':
            if m['pad'] > 0:
                layers.append(nn.ConstantPad2d(m['pad'], m['pad']))
            layers.append(
                nn.Conv2d(current_shape[0], m['filter_count'], m['filter_size'], stride=m['stride']))
            new_shape = conv_pad_shape(current_shape, m['filter_size'], m['filter_count'], m['pad'], m['stride'])
            current_shape = update_shape(f'{i}-conv', current_shape, new_shape)

            if model_params['batch_norm']:
                layers.append(nn.BatchNorm2d(current_shape[0]))
            layers.append(nn.ReLU())

        if m['type'] == 'pool':
            if m['pool_func'] == 'max':
                layers.append(nn.MaxPool2d(m['pool_size'], stride=m['stride']))
            elif m['pool_func'] == 'avg':
                layers.append(nn.AvgPool2d(m['pool_size'], stride=m['stride']))

            new_shape = pool_shape(current_shape, m['pool_size'], m['stride'])
            current_shape = update_shape(f'{i}-pool', current_shape, new_shape)


        if model_params['dropout'] > 0:
            layers.append(nn.Dropout(model_params['dropout']))

    # MLP Section
    layers.append(nn.Flatten())
    new_shape = current_shape[0] * current_shape[1] * current_shape[2]
    current_shape = update_shape(f'flatten', current_shape, new_shape)

    for i, neuron_count in enumerate(model_params['fc_layers']):
        layers.append(nn.Linear(current_shape, neuron_count))
        current_shape = update_shape(f'{i}-fc', current_shape, neuron_count)
        if model_params['batch_norm']:
            layers.append(nn.BatchNorm1d(current_shape))
        layers.append(nn.ReLU())
        if model_params['dropout'] > 0:
            layers.append(nn.Dropout(model_params['dropout']))

    # Final layer to yield classes
    layers.append(nn.Linear(current_shape, classes))
    update_shape(f'output', current_shape, classes)

    model = nn.Sequential(*layers)

    return model


def update_shape(name, current_shape, new_shape):
    print(f'{name}: {current_shape} --> {new_shape}')
    return new_shape


def validate_int(inputs: Tuple):
    if not all([isinstance(item, int) or item.is_integer() for item in inputs]):
        raise Exception(f'Not all values in list are integers: {inputs}')
    return [int(i) for i in inputs]


def conv_pad_shape(input_shape: Tuple[int, ...], filter_dims: Tuple[int, int], filter_count: int, padding=0, stride=1):
    if len(input_shape) != 3:
        raise Exception(f'Expected 3D shape, but got {input_shape} instead.')
    C, H, W = input_shape
    h_filter, w_filter = filter_dims
    H_new = (H - h_filter + 2 * padding) / stride + 1
    W_new = (W - w_filter + 2 * padding) / stride + 1

    return validate_int((filter_count, H_new, W_new))


def pool_shape(input_shape: Tuple[int, ...], pool_dims: Tuple[int, int], stride=1):
    if len(input_shape) != 3:
        raise Exception(f'Expected 3D shape, but got {input_shape} instead.')
    C, H, W = input_shape
    h_filter, w_filter = pool_dims
    H_new = (H - h_filter) / stride + 1
    W_new = (W - w_filter) / stride + 1

    return validate_int((C, H_new, W_new))

# # Does not implement stride
# def get_parameter_shape(input_shape: Tuple[int, ...], module_config: dict) -> Tuple[int, ...]:
#     if type(input_shape) is not tuple or len(input_shape) != 3:
#         raise Exception(f'Expect 3D shape, but got {input_shape} instead.')
#
#     if module_config['type'] == 'conv':
#         # (N, H, W, C) -> (N, H', W', F)
#         H, W, C = input_shape
#         HH, WW = module_config['filter_size']
#         H_s = 1 + H - HH
#         W_s = 1 + W - WW
#         F = module_config['filter_count']
#         output_shape = (H_s, W_s, F)
#         return output_shape
#
#     if module_config['type'] == 'conv-pool':
#         # (N, H, W, C) -> (N, H', W', F)
#         conv_output = get_parameter_shape(input_shape, module_config | {'type': 'conv'})
#         # (N, H', W', F) -> (N, H'', W'', F')
#         H_s, W_s, F = conv_output
#         H_ss = 1 + H_s - module_config['pool_size'][0]
#         W_ss = 1 + W_s - module_config['pool_size'][1]
#         output_shape = (H_ss, W_ss, F)
#         return output_shape
#
#     if module_config['type'] == 'pad-conv':
#         H, W, C = input_shape
#         H_p = module_config['pad'] * 2 + H
#         W_p = module_config['pad'] * 2 + W
#         padded_input = (H_p, W_p, C)
#         return get_parameter_shape(padded_input, module_config | {'type': 'conv'})
#
#     if module_config['type'] == 'pad-conv-pool':
#         H, W, C = input_shape
#         H_p = module_config['pad'] * 2 + H
#         W_p = module_config['pad'] * 2 + W
#         padded_input = (H_p, W_p, C)
#         return get_parameter_shape(padded_input, module_config | {'type': 'conv-pool'})
