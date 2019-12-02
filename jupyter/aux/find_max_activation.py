"""Find_max_activation.
"""

import numpy as np
import numpy
from numpy import linalg as LA
from collections import Counter


def _find_max_activation_v1(img_index, conv_output):
    """Find thr max activation and return its index.

    Args:
        conv_output: (1, channel, height, width)

    Returns:
        max_index(int):
    """
    conv_output = np.squeeze(conv_output[img_index])
    (channel, height, width) = conv_output.shape
    conv_output = np.reshape(conv_output, (channel, height * width))
    max_value = np.argmax(conv_output, axis=1)
    max_conv_index = np.argmax(max_value)
    return max_conv_index


def _find_max_activation_v2(img_index, conv_output):
    """Find thr max activation and return its index.
    Average over non zero pixels

    Args:
        conv_output: (1, channel, height, width)

    Returns:
        max_index(int):
    """
    conv_output = np.squeeze(conv_output[img_index])
    (channel, height, width) = conv_output.shape
    conv_output = np.reshape(conv_output, (channel, height * width))
    # mean value over non zero pixels
    conv_output_mean_row = np.true_divide(conv_output.sum(axis=1),
                                          (conv_output != 0).sum(axis=1))
    # print(conv_output_mean_row)
    max_conv_index = np.argmax(conv_output_mean_row)
    return max_conv_index


def _find_max_activation_v3(img_index, conv_output, prob=0.05):
    """Find max activation and return its index.
    Averger over top k prob(a > threshold) = 0.05
    """
    conv_output = np.squeeze(conv_output[img_index])
    (channel, height, width) = conv_output.shape
    conv_output = np.reshape(conv_output, (channel, height * width))
    top_k_num = int(height * width * prob)
    top_component = np.sort(conv_output, axis=1)[:, -top_k_num:]
    conv_output_mean_top_k = np.mean(top_component, axis=1)
    # top_component = torch.topk(conv_output, k=top_k_num)[0]
    # top_component = torch.mean(top_component, dim=1, k=top_k_num)
    # mean value over non zero pixels
    # print(conv_output_mean_row)
    max_conv_index = np.argmax(conv_output_mean_top_k)
    return max_conv_index


def _find_max_activation_v4(img_index, conv_output):
    """Find max activation and return its index.
    L2 Norm over each channel
    """
    conv_output = np.squeeze(conv_output[img_index])
    (channel, height, width) = conv_output.shape
    conv_output = np.reshape(conv_output, (channel, height * width))
    conv_output_l2 = LA.norm(conv_output, axis=1)
    # top_component = torch.topk(conv_output, k=top_k_num)[0]
    # top_component = torch.mean(top_component, dim=1, k=top_k_num)
    # mean value over non zero pixels
    # print(conv_output_mean_row)
    max_conv_index = np.argmax(conv_output_l2)
    return max_conv_index


def find_max_activation_v1(imgs_path, conv_output_indexes, conv_output):
    """find_max_activation_v1.
    Args:
        imgs_path: list
        conv_output_indexes: list
        conv_output: [batch_size, n_channels, heights, width]
    """
    # Iterate print out max filter for each images in same layer using
    # find_max_activation_v1
    for layer in range(len(conv_output_indexes)):
        print("Layer: {} ".format(layer))
        counter = Counter()
        for index in range(len(imgs_path)):
            max_filter = _find_max_activation_v1(index, conv_output[layer])
            counter.update([max_filter])
            # print(max_filter)
        print(counter)
        print("-"*80)


def find_max_activation_v2(imgs_path, conv_output_indexes, conv_output):
    # Iterate print out max filter for each images in same layer using
    # find_max_activation_v2
    for layer in range(len(conv_output_indexes)):
        print("Layer: {} ".format(layer))
        counter = Counter()
        for index in range(len(imgs_path)):
            max_filter = _find_max_activation_v2(index, conv_output[layer])
            counter.update([max_filter])
            # print(max_filter)
        print(counter)
        print("-"*80)


def find_max_activation_v3(imgs_path, conv_output_indexes, conv_output):
    prob = 0.1
    for layer in range(len(conv_output_indexes)):
        print("Layer: {} ".format(layer))
        counter = Counter()
        for index in range(len(imgs_path)):
            max_filter = _find_max_activation_v3(index, conv_output[layer],
                                                 prob=prob)
            counter.update([max_filter])
            # print(max_filter)
        print(counter)
        print("-"*80)


def find_max_activation_v4(imgs_path, conv_output_indexes, conv_output):
    for layer in range(len(conv_output_indexes)):
        print("Layer: {} ".format(layer))
        counter = Counter()
        for index in range(len(imgs_path)):
            max_filter = _find_max_activation_v4(index, conv_output[layer])
            counter.update([max_filter])
            # print(max_filter)
        print(counter)
        print("-"*80)
