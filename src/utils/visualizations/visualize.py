"""Visualize.
"""

import sys
import os
import numpy as np
from PIL import Image
import PIL
from matplotlib import pyplot as plt


sys.path.append("jupyter/")
from aux.visualization import visualize_features_map_for_comparision


def visualize(ori_activation_maps, opt_activation_maps,
              img_index=-1, layer_name="relu", backbone=None, num_class=30,
              exp=-1, imgs_path=None):
    color_map = "nipy_spectral"
    layer_index = 0
    conv_output_index_dict = {0: 0}
    index2image = {index: item.split("/")[-1].split(".")[0]
                   for index, item in enumerate(imgs_path)}
    save_dict = {
        "save_dir": "./saved/generated/" + exp + "/feature_map/",
        "index2image": index2image,
        "save_name": "layer-{}-{}-Comparision.pdf"
    }

    try:
        os.mkdir(save_dict["save_dir"])
    except FileExistsError:
        print("Directory has been created {}".format(save_dict["save_dir"]))

    if img_index != -1:
        visualize_features_map_for_comparision(
            img_index=0, layer_index=layer_index,
            features_map=ori_activation_maps,
            opt_feature_map=opt_activation_maps,
            cols=8, conv_output_index_dict=conv_output_index_dict,
            save_dict=save_dict, is_save=True,
            plt_mode="img_scale", color_map=color_map, layer_name=layer_name)
    else:
        for index in range(num_class):
            visualize_features_map_for_comparision(
                img_index=index, layer_index=layer_index,
                features_map=ori_activation_maps,
                opt_feature_map=opt_activation_maps,
                cols=8, conv_output_index_dict=conv_output_index_dict,
                save_dict=save_dict, is_save=True,
                plt_mode="img_scale", color_map=color_map,
                layer_name=layer_name)


def preprocess_arrays(original_image, opt_image,
                      ori_activation_maps, opt_activation_maps,
                      selected_filter,
                      color_map="nipy_spectral"):
    """Preprocess arrays.
    Output same shape.

    Args:
        original_image: [batch_size, 3, 224, 224]
        opt_image: [batch_size, 3, 224, 224]
        ori_activation_maps: [batch_size, channels, height, width]
        opt_activation_maps: [batch_size, channels, height, width]
        selected_filter: int
    """
    # original_image_cpu = original_image.detach().clone().cpu().numpy()
    # opt_image_cpu = opt_image.detach().clone().cpu().numpy()
    original_image_cpu = original_image.copy()
    opt_image_cpu_old = opt_image.copy()
    opt_image_cpu = opt_image.copy()
    ori_activation_maps_cpu = ori_activation_maps.copy()
    opt_activation_maps_cpu = opt_activation_maps.copy()
    # pixel_max = np.max(ori_activation_maps_cpu)
    # pixel_min = np.min(opt_activation_maps_cpu)

    ori_activation_maps_cpu = np.expand_dims(
        ori_activation_maps_cpu[:, selected_filter], axis=1)
    opt_activation_maps_cpu = np.expand_dims(
        opt_activation_maps_cpu[:, selected_filter], axis=1)

    if original_image_cpu.shape[3] != 3:
        original_image_cpu = np.transpose(original_image_cpu, (0, 2, 3, 1))

    if opt_image_cpu_old.shape[3] != 3:
        opt_image_cpu_old = np.transpose(opt_image_cpu_old, (0, 2, 3, 1))
        opt_image_cpu = np.transpose(opt_image_cpu, (0, 2, 3, 1))

    ori_activation_maps_cpu = np.transpose(ori_activation_maps_cpu,
                                           (0, 2, 3, 1))
    opt_activation_maps_cpu = np.transpose(opt_activation_maps_cpu,
                                           (0, 2, 3, 1))

    ori_activation_maps_cpu_old = ori_activation_maps_cpu.copy()
    opt_activation_maps_cpu_old = opt_activation_maps_cpu.copy()
    ori_activation_maps_cpu = []
    opt_activation_maps_cpu = []
    diff_activation_maps_cpu = []
    opt_image_cpu_scale = []

    # generate color map from gray images.
    for ori, opt, opt_scale in zip(ori_activation_maps_cpu_old,
                                   opt_activation_maps_cpu_old,
                                   opt_image_cpu_old):
        pixel_max = np.max([ori, opt])
        pixel_min = np.min([ori, opt])
        if np.max(ori) < np.max(opt):
            print("MAX: ori smaller.")
        if np.min(ori) > np.min(opt):
            print("MIN: ori bigger.")
        diff = np.abs(ori - opt)
        colored_ori = generate_colormap(ori, pixel_max, pixel_min, color_map)
        colored_opt = generate_colormap(opt, pixel_max, pixel_min, color_map)
        colored_diff = generate_colormap(diff, pixel_max, pixel_min, color_map)
        ori_activation_maps_cpu.append(colored_ori)
        opt_activation_maps_cpu.append(colored_opt)
        diff_activation_maps_cpu.append(colored_diff)

        pixel_max = np.max(opt_scale)
        pixel_min = np.min(opt_scale)
        opt_scale = (opt_scale - pixel_min) / (pixel_max - pixel_min)
        opt_image_cpu_scale.append(opt_scale)

    # ori_activation_maps_cpu = np.repeat(ori_activation_maps_cpu, 3, axis=3)
    # opt_activation_maps_cpu = np.repeat(opt_activation_maps_cpu, 3, axis=3)
    ori_activation_maps_cpu = np.squeeze(ori_activation_maps_cpu)[..., :3]
    opt_activation_maps_cpu = np.squeeze(opt_activation_maps_cpu)[..., :3]
    diff_activation_maps_cpu = np.squeeze(diff_activation_maps_cpu)[..., :3]

    original_image_cpu = (original_image_cpu * 255.0).astype(np.uint8)

    # img_max = np.max(opt_image_cpu)
    # img_min = np.min(opt_image_cpu)
    # opt_image_cpu_scale = (opt_image_cpu - img_min) / (img_max - img_min)

    opt_image_cpu = (opt_image_cpu * 255.0).astype(np.uint8)
    opt_image_cpu_scale = np.array(opt_image_cpu_scale)
    opt_image_cpu_scale = (opt_image_cpu_scale * 255.0).astype(np.uint8)
    ori_activation_maps_cpu = (ori_activation_maps_cpu * 255.0).\
        astype(np.uint8)
    opt_activation_maps_cpu = (opt_activation_maps_cpu * 255.0).\
        astype(np.uint8)
    diff_activation_maps_cpu = (diff_activation_maps_cpu * 255.0).\
        astype(np.uint8)

    return original_image_cpu, opt_image_cpu, ori_activation_maps_cpu, \
        opt_activation_maps_cpu, opt_image_cpu_scale, diff_activation_maps_cpu


def concat_imgs(original_image_cpu, opt_image_cpu,
                ori_activation_maps_cpu, opt_activation_maps_cpu,
                opt_image_cpu_scale, diff_activation_maps_cpu):
    """Concat imgs for batches.
    Args: unsigned int 8
        original_image: [batch_size, 3, height, width]
        opt_image: [batch_size, 3, height, width]
        ori_activation_maps: [batch_size, height, width]
        opt_activation_maps: [batch_size, height, width]
    """
    width = height = 224

    out_array = None

    for ori, opt, ori_ac, opt_ac, opt_scale, diff in \
            zip(original_image_cpu, opt_image_cpu, ori_activation_maps_cpu,
                opt_activation_maps_cpu, opt_image_cpu_scale,
                diff_activation_maps_cpu):
        ori_ac = Image.fromarray(ori_ac).\
            resize((width, height), PIL.Image.BICUBIC)
        opt_ac = Image.fromarray(opt_ac).\
            resize((height, width), PIL.Image.BICUBIC)
        diff = Image.fromarray(diff).\
            resize((height, width), PIL.Image.BICUBIC)

        new_cols = np.hstack((ori, opt, opt_scale, ori_ac, opt_ac, diff))
        if out_array is None:
            out_array = new_cols
        else:
            out_array = np.vstack((out_array, new_cols))
    out_array = out_array.astype(np.uint8)
    concated_imgs = Image.fromarray(out_array, mode="RGB")
    return concated_imgs


def generate_colormap(featureMap, pixel_max, pixel_min,
                      color_map="nipy_spectral"):
    """Generate colormap.
    """
    cm = plt.get_cmap(color_map)
    featureMap = (featureMap - pixel_min) / (pixel_max - pixel_min)
    colored_featureMap = cm(featureMap)
    return colored_featureMap
