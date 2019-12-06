"""Visualization method.
"""


import os
import copy
import math
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import PIL
from PIL import Image
import numpy as np

from .utils import cat_img_horizontal


def visualize_features_map(img_index: int, layer_index: int, features_map,
                           opt_feature_map, cols=4,
                           conv_output_index_dict=None,
                           save_dict=None):
    """Visulize feature map.
    Args:
        img_index:
        layer_index: absolute layer index start from 0
    """
    list_index = conv_output_index_dict[layer_index]
    features_map = copy.deepcopy(features_map)[list_index]
    opt_features_map = copy.deepcopy(opt_feature_map)[list_index]
    assert features_map.shape == opt_features_map.shape
    features_map = features_map.transpose((0, 2, 3, 1))
    opt_features_map = opt_features_map.transpose((0, 2, 3, 1))
    n_filters = features_map.shape[-1]
    rows = math.ceil(n_filters / cols)

    fig = plt.figure(constrained_layout=True)
    fig_width = 8+1
    fig_height = int(rows*0.9)
    fig = plt.figure(constrained_layout=False, figsize=(fig_width, fig_height))
    gs = GridSpec(rows, cols*2, figure=fig)
    gs.update(wspace=0.025, hspace=0.1)

    font = {'family': 'normal', 'size': 4}
    width = height = 224
    index = 0
    for row in range(0, rows):
        for col in range(0, cols):
            # specify subplot and turn off axis
            ax = fig.add_subplot(gs[row, col*2: col*2+2])
            ax.set_xticks([])
            ax.set_yticks([])
            # plot feature maps in grayscale
            img = Image.fromarray(features_map[img_index, :, :, index]).\
                resize((width, height), PIL.Image.BICUBIC)
            opt_img = Image.fromarray(
                opt_features_map[img_index, :, :, index]).\
                resize((width, height), PIL.Image.BICUBIC)
            cat_img = cat_img_horizontal(img, opt_img)
            cat_img_np = np.array(cat_img)
            plt.imshow(cat_img_np, cmap="gray")
            ax.set_title("{}.".format(index), loc="center", pad=1.0,
                         fontdict=font)
            index += 1
    # show figure
    # fig.suptitle("Layer-{}".format(layer_index), fontsize=8,
    #              verticalalignment="bottom")
    # plt.tight_layout()
    file_name = os.path.join(save_dict["save_dir"], save_dict["save_name"].
                             format(layer_index, save_dict["index2image"]
                                    [img_index]))
    plt.savefig(file_name)
    print("Successfully Save pdf to {}".format(file_name))
    plt.show()


def visualize_filters(filters, n_filters, n_channels):
    """Visualize filters.
    Args:
        filters: [n_channels, rows, cols]
        n_filters: int, numbers of filters.
        n_channels: int, numbers of channels for each filter.
    """
    index = 1
    # get the feature map of specify channels
    for row in range(n_filters):
        fs = filters[:, :, :, row]
        for col in range(n_channels):
            # specify subplot and turn off axis
            ax = plt.subplot(n_filters, n_channels, index)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in gray scale
            plt.imshow(fs[:, :, col], cmap="gray")
            index += 1
    # show the figure
    plt.show()
