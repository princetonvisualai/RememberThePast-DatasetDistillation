import torch
import torch.optim as optim


"""
  Get optimizer for a model
"""
def get_optimizer(parameters, optim_config):
    if optim_config.optimizer == 'AdamW':
        return optim.AdamW(parameters, lr=optim_config.lr, weight_decay=float(optim_config.weight_decay))
    elif optim_config.optimizer == 'Adam':
        return optim.Adam(parameters, lr=optim_config.lr, weight_decay=float(optim_config.weight_decay))
    elif optim_config.optimizer == 'SGD':
        return optim.SGD(parameters, lr=optim_config.lr, momentum=optim_config.momentum)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(optim_config.optimizer))
