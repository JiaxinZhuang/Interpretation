"""Utils.py
    Functions for visualization
"""

import os
import sys
import math
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score


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


def obtain_features_map(image, model, layer_output_indexes=None,
                        order=[0, 1, 2]):
    """Obtain feature map
    Args:
        image: [batch_size, n_channels, height, width]
        model: net
        layer_output_indexes: list, contains selected indedx for layer.
        order: [0, 1, 2]
    Return:
    """
    layer_output = []
    layer_max_min = []
    out = image[:, order]

    model.eval()
    with torch.no_grad():
        for index, layer in enumerate(model):
            out = layer(out)
            # print(index, layer, out.size())
            if index in layer_output_indexes:
                out_np = out.cpu().detach().numpy()
                layer_output.append(out_np)

                layer_min = np.min(out_np)
                layer_max = np.max(out_np)
                layer_max_min.append([layer_max, layer_min])
                # print("Index:{}, {}".format(index, layer))
                # print(np.min(out_np), np.max(out_np))
    return layer_output, layer_max_min


def load_imgs(ab_path: str, imgs_path: list, non_exists_ok=False):
    """Load imgs.
    Args:
        ab_path: path to saved generated images.
        imgs_path: list, containing many unrelative prefix path.
        non_exists_ok: Boolean, imgs_path may not exist in ab_path, True is to
                       allow such situation.
    """
    try:
        existed_imgs = os.listdir(ab_path)
    except Exception:
        print("FileNotFoundError: {}".format(ab_path))
        return
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


def get_gt_pred_specific_class(net, trainloader, order=[0, 1, 2], device=None):
    y_true = []
    y_pred = []
    net.eval()
    with torch.no_grad():
        for index, (data, target, *_) in enumerate(trainloader):
            if order is not None:
                data = data[:, order]
            data = data.to(device)
            predict = torch.argmax(net(data), dim=1).cpu().data.numpy()
            y_pred.extend(predict)
            target = target.cpu().data.numpy()
            y_true.extend(target)
    acc = accuracy_score(y_true, y_pred)
    del data
    return y_true, y_pred, acc


def get_differ(origin, changed, top=100):
    origin_acc = np.array([acc for *_, acc in origin.values()])
    changed_acc = np.array([acc for *_, acc in changed.values()])
    differ = origin_acc - changed_acc
    arg_index = np.argsort(differ)
    return arg_index, differ, origin_acc, changed_acc


def fill_AArray_under_resolution(value, width, height):
    """Fill a 2D array with a specific value under wdith x height.
    """
    aCol = np.repeat(value, height, axis=0)
    array = np.repeat([aCol], width, axis=0)
    return array


def get_rgb_points_by_batch(batch_size=96, width=224, height=224, step=8):
    """Get batch by batch size with step.
    """
    red_max = 256
    green_max = 256
    blue_max = 256
    counter = 0
    imgs = []
    dims = ((red_max - 1) / step + 1) ** 3
    last_iteration = dims
    xs = []
    ys = []
    zs = []

    for R in range(0, red_max, step):
        for G in range(0, green_max, step):
            for B in range(0, blue_max, step):
                rs = fill_AArray_under_resolution(R, width, height)
                gs = fill_AArray_under_resolution(G, width, height)
                bs = fill_AArray_under_resolution(B, width, height)
                imgs.append(np.dstack((rs, gs, bs)))
                xs.append(R)
                ys.append(G)
                zs.append(B)
                counter += 1
                if counter % batch_size == 0 or counter == last_iteration:
                    imgs = np.array(imgs, dtype=np.float32)
                    yield imgs, xs, ys, zs
                    del imgs, xs, ys, zs
                    imgs = []
                    xs = []
                    ys = []
                    zs = []


def obtain_selected_4D_featureMap(net=None,
                                  layer_output_indexes=None,
                                  selected_filter=None,
                                  batch_size=96, step=8,
                                  method="max", device=None):
    """Args:
        method: [max, median, mean]
    """
    rets = []
    if method == "max":
        sel = np.max
    elif method == "median":
        sel = np.median
    elif method == "mean":
        sel = np.mean
    else:
        print("No method")
        sys.exit(-1)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    dataloader = get_rgb_points_by_batch(batch_size=batch_size, step=step)
    all_iterations = math.ceil((255 / step + 1) ** 3 / batch_size)
    # xs => r, gs => g, zs => b
    xs = []
    ys = []
    zs = []
    ret = []
    for index, (imgs, x, y, z) in enumerate(dataloader):
        if index % 10 == 0:
            print("[{}/{}]".format(index, all_iterations))
        xs.extend(x)
        ys.extend(y)
        zs.extend(z)
        data = zscore(imgs, mean, std)
        data = torch.tensor(data).to(device)
        layer_output, _ = obtain_features_map(
            data, net.model.features,
            layer_output_indexes=layer_output_indexes)
        ret = sel(layer_output[0][:, selected_filter], axis=(1, 2))
        rets.extend(ret)
        del layer_output
    del data
    rets = np.array(rets)
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    zs = np.array(zs, dtype=np.float32)
    return xs, ys, zs, rets
