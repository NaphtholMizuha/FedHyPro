import numpy as np
import torch

def state_dict_to_flatten(state_dict: dict) -> np.ndarray:
    """
    Flatten the state_dict of the model
    """
    flattened = np.array([])
    for _, param in state_dict.items():
        param_flat = param.cpu().numpy().flatten()
        flattened = np.concatenate((flattened, param_flat))
    return flattened.copy()

def flatten_to_state_dict(src: np.ndarray, template: dict) -> dict:
    """
    Recover the state_dict of the model from the flattened update
    """
    state_dict = {}
    cur_idx = 0
    for name, param in template.items():
        param_flat = src[cur_idx:cur_idx + param.numel()]
        state_dict[name] = torch.from_numpy(param_flat.reshape(param.shape))
        cur_idx += param.numel()
    return state_dict.copy()

def quantize(state_dict: dict, qparam) -> dict:
    for key, tensor in state_dict.items():
        min_val, scale = qparam[key]
        state_dict[key] = ((tensor - min_val) / scale).round().clamp(0, 65535)
    grad = state_dict_to_flatten(state_dict).astype(np.uint16)
    return grad
        
def unquantize(grad, template: dict, qparam) -> dict:
    grad = grad.astype(np.int32)
    state_dict = flatten_to_state_dict(grad, template)
    for key, tensor in state_dict.items():
        min_val, scale = qparam[key]
        min_val = min_val.to('cpu')
        scale = scale.to('cpu')
        state_dict[key] = tensor.to(torch.float32) * scale + min_val
    return state_dict