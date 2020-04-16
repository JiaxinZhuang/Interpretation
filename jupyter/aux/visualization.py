"""Visualization method.
"""


import os
import sys
import copy
import math
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import PIL
from PIL import Image
import numpy as np
import heapq

from .utils import cat_img_horizontal


def visualize_features_map_for_comparision(img_index: int, layer_index:
                                           int, features_map,
                                           opt_feature_map, cols=4,
                                           conv_output_index_dict=None,
                                           save_dict=None, plt_mode="real",
                                           top_k=10, layer_max_min=None,
                                           color_map="gray"):
    """Visulize feature map for comparision!!
    Args:
        img_index:
        layer_index: absolute layer index start from 0
        plt_mode: [real, img_scale, imgs_scale]
        color_map: [gray, jet, ...], choices available on
            https://matplotlib.org/3.1.3/tutorials/colors/colormaps.html
    """
    print("Plot mode is => {}".format(plt_mode))
    print("Color map is => {}".format(color_map))
    list_index = conv_output_index_dict[layer_index]
    features_map = copy.deepcopy(features_map)[list_index]
    opt_features_map = copy.deepcopy(opt_feature_map)[list_index]
    assert features_map.shape == opt_features_map.shape
    features_map = features_map.transpose((0, 2, 3, 1))
    opt_features_map = opt_features_map.transpose((0, 2, 3, 1))
    n_filters = features_map.shape[-1]
    rows = math.ceil(n_filters / cols)

    # fig = plt.figure(constrained_layout=True)
    fig_width = 8+1
    fig_height = int(rows*0.8)
    fig = plt.figure(constrained_layout=False, figsize=(fig_width, fig_height))
    gs = GridSpec(rows, int(cols * 2), figure=fig)
    gs.update(wspace=0.025, hspace=0.1)

    font = {'family': 'normal', 'size': 4}
    width = height = 224
    index = 0
    layer_max = np.max(features_map[img_index, :, :, :])
    layer_min = np.min(features_map[img_index, :, :, :])
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

            max_pixel_gt = np.max(img)
            min_pixel_gt = np.min(img)
            max_pixel_opt = np.max(opt_img)
            min_pixel_opt = np.min(opt_img)

            if plt_mode == "real":
                plt_show(cat_img_np, plt_mode=plt_mode, color_map=color_map)
            elif plt_mode == "img_scale":
                pixel_max = layer_max
                pixel_min = layer_min
                plt_show(cat_img_np, plt_mode=plt_mode, pixel_max=pixel_max,
                         pixel_min=pixel_min, color_map=color_map)
            elif plt_mode == "imgs_scale":
                print("TODO")
                sys.exit(-1)
                # pixel_max = layer_max_min[list_index][1]
                # pixel_min = layer_max_min[list_index][0]
                # plt_show(cat_img_np, plt_mode=plt_mode, pixel_max=pixel_max,
                #          pixel_min=pixel_min)
            else:
                sys.exit(-1)

            # ax.set_title("{}.".format(index), loc="center", pad=1.0,
            #              fontdict=font)

            ax.set_title("{}--GT-[{:.1f}~{:.1f}]-[{:.1f}~{:.1f}]".\
                         format(index, min_pixel_gt, max_pixel_gt,
                                min_pixel_opt, max_pixel_opt),
                         loc="center", pad=1.0, fontdict=font)
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
    # plt.show()


def visualize_features_map(img_index: int, layer_index: int, features_map,
                           cols=8, conv_output_index_dict=None,
                           save_dict=None, is_save=False,
                           save_original=False, plt_mode="real", top_k=10,
                           layer_max_min=None, color_map="gray"):
    """Visualize feature map.
    Args:
        img_index:
        layer_index: absolute layer index start from 0
        is_save: save figures to pdf
        save_original: save imgs under a directory.
        plt_mode: real, img_scale, imgs_scale
        color_map: [gray, jet, ...], choices available on
            https://matplotlib.org/3.1.3/tutorials/colors/colormaps.html
    """
    print("Plot mode is => {}".format(plt_mode))
    print("Color map is => {}".format(color_map))
    list_index = conv_output_index_dict[layer_index]
    features_map = copy.deepcopy(features_map)[list_index]
    features_map = features_map.transpose((0, 2, 3, 1))
    n_filters = features_map.shape[-1]
    rows = math.ceil(n_filters / cols)

    fig = plt.figure(constrained_layout=True)
    fig_width = 8+1
    fig_height = int(rows*0.9)
    fig = plt.figure(constrained_layout=False, figsize=(fig_width, fig_height))
    gs = GridSpec(rows, cols, figure=fig)
    gs.update(wspace=0.025, hspace=0.1)

    font = {'family': 'normal', 'size': 4}
    width = height = 224

    index = 0
    max_values = {}
    layer_max = np.max(features_map[img_index, :, :, :])
    layer_min = np.min(features_map[img_index, :, :, :])
    for row in range(0, rows):
        for col in range(0, cols):
            # specify subplot and turn off axis
            ax = fig.add_subplot(gs[row, col: col+1])
            ax.set_xticks([])
            ax.set_yticks([])
            # plot feature maps in grayscale
            img = Image.fromarray(features_map[img_index, :, :, index]).\
                resize((width, height), PIL.Image.BICUBIC)
            cat_img_np = np.array(img)
            max_pixel = np.max(cat_img_np)
            min_pixel = np.min(cat_img_np)
            max_values[index] = max_pixel

            if plt_mode == "real":
                plt_show(cat_img_np, plt_mode=plt_mode, color_map=color_map)
            elif plt_mode == "img_scale":
                pixel_max = layer_max
                pixel_min = layer_min
                plt_show(cat_img_np, plt_mode=plt_mode, pixel_max=pixel_max,
                         pixel_min=pixel_min, color_map=color_map)
            elif plt_mode == "imgs_scale":
                pixel_max = layer_max_min[list_index][0]
                pixel_min = layer_max_min[list_index][1]
                plt_show(cat_img_np, plt_mode=plt_mode, pixel_max=pixel_max,
                         pixel_min=pixel_min, color_map=color_map)
            else:
                sys.exit(-1)
            ax.set_title("{}--[{:.1f}~{:.1f}]".format(index, min_pixel,
                                                      max_pixel),
                         loc="center", pad=1.0, fontdict=font)
            index += 1
    # show figure
    top_k_key = heapq.nlargest(top_k, max_values, key=max_values.get)
    print("Top-k => {}".format(top_k_key))
    fig.suptitle("Image-{}-Layer-{}--[{:.1f}-{:.1f}]".format(img_index,
                 layer_index, layer_min, layer_max), fontsize=8)
    # plt.tight_layout()
    if is_save:
        file_name = os.path.join(save_dict["save_dir"], save_dict["save_name"].
                                 format(layer_index, save_dict["index2image"]
                                        [img_index]))
        plt.savefig(file_name)
        print("Successfully Save pdf to {}".format(file_name))
    plt.show()

    if save_original:
        save_name = save_dict["save_name"].split(".")[0]
        directory = os.path.join(save_dict["save_dir"], save_name.format(
            layer_index, save_dict["index2image"][img_index]
        ))
        save_under_directory(features_map[img_index, :, :, :], directory)


def save_under_directory(img_arr, path):
    """Save img under directory.
    """
    try:
        os.mkdir(path)
    except FileExistsError:
        print("Directory has been created {}".format(path))
    num_imgs = img_arr.shape[2]
    for index in range(num_imgs):
        img = Image.fromarray(img_arr[:, :, index]).convert("L")
        save_path = os.path.join(path, "{}.bmp".format(index))
        img.save(save_path)


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
    # plt.show()


def plt_show(cat_img_np, plt_mode="real", pixel_max=None, pixel_min=None,
             color_map=None):
    """Plt under the plt mode.
    Args:
        plt_mode: [real, img_scale, imgs_scale]
    """
    if plt_mode == "real":
        plt.imshow(cat_img_np, cmap=color_map, vmin=0, vmax=255)
    elif plt_mode == "img_scale":
        cat_img_np = (cat_img_np - pixel_min) / (pixel_max - pixel_min) * 255
        plt.imshow(cat_img_np, cmap=color_map, vmin=0, vmax=255)
    elif plt_mode == "imgs_scale":
        cat_img_np = (cat_img_np - pixel_min) / (pixel_max - pixel_min) * 255
        plt.imshow(cat_img_np, cmap=color_map, vmin=0, vmax=255)
    else:
        sys.exit(-1)
