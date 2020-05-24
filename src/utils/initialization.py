"""Initialization.
"""

import torch.nn as nn


def _xavier_uniform(net, _print):
    """Xavier normalization.
    """
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            _print("Init Conv2d")
            nn.init.xavier_uniform_(module.weight,
                                    gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            _print("Init Batch-normalization")
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            _print("Init Linear")
            nn.init.xavier_uniform_(module.weight,
                                    gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    return net


def _xavier_normal(net, _print):
    """Xavier normalization.
    """
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            _print("Init Conv2d")
            nn.init.xavier_normal_(module.weight,
                                   gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            _print("Init Batch-normalization")
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            _print("Init Linear")
            nn.init.xavier_normal_(module.weight,
                                   gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    return net


def _kaiming_normal(net, _print):
    """Kaiming normalization.
    """
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            _print("Init Conv2d")
            nn.init.kaiming_normal_(module.weight, mode='fan_in',
                                    nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            _print("Init Batch-normalization")
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            _print("Init Linear")
            nn.init.kaiming_normal_(module.weight, mode='fan_in',
                                    nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.ConvTranspose2d):
            _print("Init Conv2dT")
            nn.init.kaiming_normal_(module.weight, mode='fan_in',
                                    nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    return net


def _kaiming_uniform(net, _print):
    """Kaiming normalization.
    """
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            _print("Init Conv2d")
            nn.init.kaiming_uniform_(module.weight, mode='fan_in',
                                     nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            _print("Init Batch-normalization")
            nn.init.kaiming_uniform_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            _print("Init Linear")
            nn.init.kaiming_uniform_(module.weight, mode='fan_in',
                                     nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    return net
