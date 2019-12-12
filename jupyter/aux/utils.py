"""Utils.py
    Functions for visualization
"""

import os
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def print_ckpt(ckpt):
    """Print the size of tensor for conv & linear & bn in ckpt.
    Args:
        ckpt: dicts for pytorch
    Returns:
        params: dict, contains conv, linear, bn
        params_name: list, layers' name
    """
    params = {}
    params_name = []
    index = 1
    for name, param in ckpt.items():
        if "bias" in name:
            params[index] = param.cpu().numpy()
            params_name.append(name)
            print(index, name, param.size())
            index += 1
    print("params has len {}".format(len(params)))
    return params, params_name


def print_layer(net):
    """Print layer with index.
    Args:
        model
    """
    for index, layer in enumerate(net.model.features):
        print(index, layer)


def plot_bias(params, params_name):
    """Plot bias for conv, linear & bn in seperate figure.
    Args:
        params: dict
        params_name: list
    """
    # plot the bias
    cols = 2
    rows = int(len(params) / cols)
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)

    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            axes[row, col].hist(params[index+1], bins=150, color="k",
                                alpha=0.5, label=index)
            axes[row, col].set_xlim(-1, 1)
            axes[row, col].set_xticks([-1, 0, 1])
            axes[row, col].set_ylim(0, 15)
            axes[row, col].set_title(params_name[index], fontsize=4)

    matplotlib.rc('xtick', labelsize=4)
    matplotlib.rc('ytick', labelsize=4)
    plt.subplots_adjust(wspace=0, hspace=0.6)


def plot_bias_asawhole(net):
    """Plot bias for conv, linear & bn as a whole figure.
    Args:
        net: model
    """
    for name, param in net.named_parameters():
        if "bias" in name:
            param = param.clone().detach().cpu().numpy()
            ax = sns.distplot(param, label=name, kde=False)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    plt.legend(fontsize="xx-small")


def obtain_features_map(image, model, conv_output_indexes=None):
    """Obtain feature map
    Args:
        image: [batch_size, n_channels, height, width]
        model: net
        conv_output_indexes: list, contains selected indedx for layer.
    Return:
    """
    conv_output = []
    out = image
    for index, layer in enumerate(model):
        out = layer(out)
        if index in conv_output_indexes:
            conv_output.append(out.cpu().detach().numpy())
            # print(index, out.size())
    return conv_output


def load_imgs(ab_path: str, imgs_path: list, non_exists_ok=False):
    """Load imgs.
    Args:
        ab_path: path to saved generated images.
        imgs_path: list, containing many unrelative prefix path.
        non_exists_ok: Boolean, imgs_path may not exist in ab_path, True is to
                       allow such situation.
    """
    existed_imgs = os.listdir(ab_path)
    out = []
    valid_imgs_path = []
    valid_imgs_index = []
    for img_path in imgs_path:
        file_name = img_path.split("/")[-1]
        if non_exists_ok and file_name not in existed_imgs:
            valid_imgs_index.append(0)
            print("Skip {}".format(img_path))
            continue
        file_path = os.path.join(ab_path, file_name)
        print("Load from {}".format(file_path))
        img = np.array(Image.open(file_path).convert("RGB")).astype("float32")
        img = img / 255.0
        out.append(img)
        valid_imgs_index.append(1)
        valid_imgs_path.append(img_path)
    out = np.array(out)
    return out, valid_imgs_path, valid_imgs_index


def extract_valid(images: list, labels: list, valid_imgs_index: list):
    """Extract valid paths.
    Args:
        valid_imgs_path: list
    Return:
        images: list
        labels: list
    """
    valid_images = []
    valid_labels = []

    for img, label, valid_img_index in zip(images, labels, valid_imgs_index):
        if valid_img_index:
            valid_images.append(img)
            valid_labels.append(label)
    return valid_images, valid_labels


def zscore(optimized_data, mean, std):
    """Zscore the data.
    Args:
        optimized_data: [batch_size, height, width, channels]
        mean: [channels]
        std: [channels]
    Return:
        optimized_data: [batch_size, channels, heights, width]
    """
    optimized_data = np.transpose(optimized_data.copy(), (0, 3, 1, 2))
    channels = optimized_data.shape[1]
    for channel in range(channels):
        optimized_data[:, channel, :, :] = optimized_data[:, channel, :, :] -\
            mean[channel]
        optimized_data[:, channel, :, :] = optimized_data[:, channel, :, :] /\
            std[channel]
    return optimized_data


def cat_img_horizontal(img_left, img_right):
    """Concat image horizontal.
    """
    width, height = img_left.size
    out = Image.new("F", (width*2, height))
    out.paste(img_left, (0, 0))
    out.paste(img_right, (width, 0))
    return out
