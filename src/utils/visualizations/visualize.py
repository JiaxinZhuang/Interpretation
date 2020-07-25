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
    #color_map = "nipy_spectral"
    color_map = "jet"
    layer_index = 0
    conv_output_index_dict = {0: 0}
    index2image = {index: item.split("/")[-1].split(".")[0]
                   for index, item in enumerate(imgs_path)}
    save_dict = {
        "save_dir": "./saved/generated/" + exp + "/feature_map/",
        "index2image": index2image,
        "save_name": "layer-{}-{}-Comparision.png"
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

    ori_activation_maps_cpu_n1e1 = []
    opt_activation_maps_cpu_n1e1 = []
    ori_activation_maps_cpu_n2e1 = []
    opt_activation_maps_cpu_n2e1 = []
    ori_activation_maps_cpu_n4e1 = []
    opt_activation_maps_cpu_n4e1 = []
    diff_activation_maps_cpu_n1e1 = []
    diff_activation_maps_cpu_n2e1 = []
    diff_activation_maps_cpu_n4e1 = []

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
        if pixel_min > 0:
            print("pixel min is not 0.")
        # threshold to feature map, 1e-1, 2e-1, 4e-1
        # 1e-1
        ori_n1e1 = ori.copy()
        ori_n1e1[ori_n1e1 < 1e-1] = 0
        opt_n1e1 = opt.copy()
        opt_n1e1[opt_n1e1 < 1e-1] = 0
        diff_n1e1 = np.abs(ori_n1e1 - opt_n1e1)
        # #Feature map colored.
        colored_ori_n1e1 = generate_colormap(ori_n1e1, pixel_max, pixel_min,
                                             color_map)
        colored_opt_n1e1 = generate_colormap(opt_n1e1, pixel_max, pixel_min,
                                             color_map)
        colored_diff_n1e1 = generate_colormap(diff_n1e1, pixel_max, pixel_min,
                                              color_map)
        # 1e-1
        ori_n2e1 = ori.copy()
        ori_n2e1[ori_n2e1 < 1e-0] = 0
        opt_n2e1 = opt.copy()
        opt_n2e1[opt_n2e1 < 1e-0] = 0
        diff_n2e1 = np.abs(ori_n2e1 - opt_n2e1)
        # #Feature map colored.
        colored_ori_n2e1 = generate_colormap(ori_n2e1, pixel_max, pixel_min,
                                             color_map)
        colored_opt_n2e1 = generate_colormap(opt_n2e1, pixel_max, pixel_min,
                                             color_map)
        colored_diff_n2e1 = generate_colormap(diff_n2e1, pixel_max, pixel_min,
                                              color_map)
        # 4e-1
        ori_n4e1 = ori.copy()
        ori_n4e1[ori_n4e1 < 2e-0] = 0
        opt_n4e1 = opt.copy()
        opt_n4e1[opt_n4e1 < 2e-0] = 0
        diff_n4e1 = np.abs(ori_n4e1 - opt_n4e1)
        # #Feature map colored.
        colored_ori_n4e1 = generate_colormap(ori_n4e1, pixel_max, pixel_min,
                                             color_map)
        colored_opt_n4e1 = generate_colormap(opt_n4e1, pixel_max, pixel_min,
                                             color_map)
        colored_diff_n4e1 = generate_colormap(diff_n4e1, pixel_max, pixel_min,
                                              color_map)
        # Package.
        ori_activation_maps_cpu_n1e1.append(colored_ori_n1e1)
        opt_activation_maps_cpu_n1e1.append(colored_opt_n1e1)
        diff_activation_maps_cpu_n1e1.append(colored_diff_n1e1)
        ori_activation_maps_cpu_n2e1.append(colored_ori_n2e1)
        opt_activation_maps_cpu_n2e1.append(colored_opt_n2e1)
        diff_activation_maps_cpu_n2e1.append(colored_diff_n2e1)
        ori_activation_maps_cpu_n4e1.append(colored_ori_n4e1)
        opt_activation_maps_cpu_n4e1.append(colored_opt_n4e1)
        diff_activation_maps_cpu_n4e1.append(colored_diff_n4e1)

        # Feature map colored
        diff = np.abs(ori - opt)
        colored_ori = generate_colormap(ori, pixel_max, pixel_min, color_map)
        colored_opt = generate_colormap(opt, pixel_max, pixel_min, color_map)
        colored_diff = generate_colormap(diff, pixel_max, pixel_min, color_map)
        # Package.
        ori_activation_maps_cpu.append(colored_ori)
        opt_activation_maps_cpu.append(colored_opt)
        diff_activation_maps_cpu.append(colored_diff)
        # Scale optimized image for special looking.
        pixel_max = np.max(opt_scale)
        pixel_min = np.min(opt_scale)
        opt_scale = (opt_scale - pixel_min) / (pixel_max - pixel_min)
        opt_image_cpu_scale.append(opt_scale)

    ori_activation_maps_cpu = np.squeeze(ori_activation_maps_cpu)[..., :3]
    opt_activation_maps_cpu = np.squeeze(opt_activation_maps_cpu)[..., :3]
    diff_activation_maps_cpu = np.squeeze(diff_activation_maps_cpu)[..., :3]
    ori_activation_maps_cpu_n1e1 = \
        np.squeeze(ori_activation_maps_cpu_n1e1)[..., :3]
    opt_activation_maps_cpu_n1e1 = \
        np.squeeze(opt_activation_maps_cpu_n1e1)[..., :3]
    diff_activation_maps_cpu_n1e1 = \
        np.squeeze(diff_activation_maps_cpu_n1e1)[..., :3]
    ori_activation_maps_cpu_n2e1 = \
        np.squeeze(ori_activation_maps_cpu_n2e1)[..., :3]
    opt_activation_maps_cpu_n2e1 = \
        np.squeeze(opt_activation_maps_cpu_n2e1)[..., :3]
    diff_activation_maps_cpu_n2e1 = \
        np.squeeze(diff_activation_maps_cpu_n2e1)[..., :3]
    ori_activation_maps_cpu_n4e1 = \
        np.squeeze(ori_activation_maps_cpu_n4e1)[..., :3]
    opt_activation_maps_cpu_n4e1 = \
        np.squeeze(opt_activation_maps_cpu_n4e1)[..., :3]
    diff_activation_maps_cpu_n4e1 = \
        np.squeeze(diff_activation_maps_cpu_n4e1)[..., :3]

    original_image_cpu = (original_image_cpu * 255.0).astype(np.uint8)

    # Scale from [0, 1] with colored to [0, 255] for output
    opt_image_cpu = (opt_image_cpu * 255.0).astype(np.uint8)
    opt_image_cpu_scale = np.array(opt_image_cpu_scale)
    opt_image_cpu_scale = (opt_image_cpu_scale * 255.0).astype(np.uint8)
    ori_activation_maps_cpu = (ori_activation_maps_cpu * 255.0).\
        astype(np.uint8)
    opt_activation_maps_cpu = (opt_activation_maps_cpu * 255.0).\
        astype(np.uint8)
    diff_activation_maps_cpu = (diff_activation_maps_cpu * 255.0).\
        astype(np.uint8)
    ori_activation_maps_cpu_n1e1 = (ori_activation_maps_cpu_n1e1 * 255.0).\
        astype(np.uint8)
    opt_activation_maps_cpu_n1e1 = (opt_activation_maps_cpu_n1e1 * 255.0).\
        astype(np.uint8)
    diff_activation_maps_cpu_n1e1 = (diff_activation_maps_cpu_n1e1 * 255.0).\
        astype(np.uint8)
    ori_activation_maps_cpu_n2e1 = (ori_activation_maps_cpu_n2e1 * 255.0).\
        astype(np.uint8)
    opt_activation_maps_cpu_n2e1 = (opt_activation_maps_cpu_n2e1 * 255.0).\
        astype(np.uint8)
    diff_activation_maps_cpu_n2e1 = (diff_activation_maps_cpu_n2e1 * 255.0).\
        astype(np.uint8)
    ori_activation_maps_cpu_n4e1 = (ori_activation_maps_cpu_n4e1 * 255.0).\
        astype(np.uint8)
    opt_activation_maps_cpu_n4e1 = (opt_activation_maps_cpu_n4e1 * 255.0).\
        astype(np.uint8)
    diff_activation_maps_cpu_n4e1 = (diff_activation_maps_cpu_n4e1 * 255.0).\
        astype(np.uint8)

    return original_image_cpu, opt_image_cpu, ori_activation_maps_cpu, \
        opt_activation_maps_cpu, opt_image_cpu_scale, \
        diff_activation_maps_cpu,\
        ori_activation_maps_cpu_n1e1, opt_activation_maps_cpu_n1e1, \
        diff_activation_maps_cpu_n1e1,\
        ori_activation_maps_cpu_n2e1, opt_activation_maps_cpu_n2e1, \
        diff_activation_maps_cpu_n2e1,\
        ori_activation_maps_cpu_n4e1, opt_activation_maps_cpu_n4e1,\
        diff_activation_maps_cpu_n4e1


def concat_imgs(original_image_cpu, opt_image_cpu,
                ori_activation_maps_cpu, opt_activation_maps_cpu,
                opt_image_cpu_scale, diff_activation_maps_cpu,
                ori_activation_maps_cpu_n1e1, opt_activation_maps_cpu_n1e1,
                diff_activation_maps_cpu_n1e1,
                ori_activation_maps_cpu_n2e1, opt_activation_maps_cpu_n2e1,
                diff_activation_maps_cpu_n2e1,
                ori_activation_maps_cpu_n4e1, opt_activation_maps_cpu_n4e1,
                diff_activation_maps_cpu_n4e1):
    """Concat imgs for batches.
    Args: unsigned int 8
        original_image: [batch_size, 3, height, width]
        opt_image: [batch_size, 3, height, width]
        ori_activation_maps: [batch_size, height, width]
        opt_activation_maps: [batch_size, height, width]
    """
    width = height = 224

    # ori, opt, opt_scale, or_fm, opt_fm, diff_fm
    out_array = None
    not_resize = None
    for ori, opt, ori_ac, opt_ac, opt_scale, diff in \
            zip(original_image_cpu, opt_image_cpu, ori_activation_maps_cpu,
                opt_activation_maps_cpu, opt_image_cpu_scale,
                diff_activation_maps_cpu):
        new_col_not_resize = np.hstack((ori_ac, opt_ac))

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

        if not_resize is None:
            not_resize = new_col_not_resize
        else:
            not_resize = np.vstack((not_resize, new_col_not_resize))

    out_array = out_array.astype(np.uint8)
    concated_imgs = Image.fromarray(out_array, mode="RGB")

    out_not_resize = not_resize.astype(np.uint8)
    out_not_resize_img = Image.fromarray(out_not_resize, mode="RGB")

    # concated feature maps.
    out_array_fms = None
    for ori_n1e1, opt_n1e1, diff_n1e1, ori_n2e1, opt_n2e1, diff_n2e1,\
            ori_n4e1, opt_n4e1, diff_n4e1 in \
            zip(ori_activation_maps_cpu_n1e1, opt_activation_maps_cpu_n1e1,
                diff_activation_maps_cpu_n1e1,
                ori_activation_maps_cpu_n2e1, opt_activation_maps_cpu_n2e1,
                diff_activation_maps_cpu_n2e1,
                ori_activation_maps_cpu_n4e1, opt_activation_maps_cpu_n4e1,
                diff_activation_maps_cpu_n4e1):
        ori_n1e1 = Image.fromarray(ori_n1e1).\
            resize((width, height), PIL.Image.BICUBIC)
        opt_n1e1 = Image.fromarray(opt_n1e1).\
            resize((height, width), PIL.Image.BICUBIC)
        diff_n1e1 = Image.fromarray(diff_n1e1).\
            resize((height, width), PIL.Image.BICUBIC)
        ori_n2e1 = Image.fromarray(ori_n2e1).\
            resize((width, height), PIL.Image.BICUBIC)
        opt_n2e1 = Image.fromarray(opt_n2e1).\
            resize((height, width), PIL.Image.BICUBIC)
        diff_n2e1 = Image.fromarray(diff_n2e1).\
            resize((height, width), PIL.Image.BICUBIC)
        ori_n4e1 = Image.fromarray(ori_n4e1).\
            resize((width, height), PIL.Image.BICUBIC)
        opt_n4e1 = Image.fromarray(opt_n4e1).\
            resize((height, width), PIL.Image.BICUBIC)
        diff_n4e1 = Image.fromarray(diff_n4e1).\
            resize((height, width), PIL.Image.BICUBIC)

        new_cols = np.hstack((
            ori_n1e1, opt_n1e1, diff_n1e1,
            ori_n2e1, opt_n2e1, diff_n2e1,
            ori_n4e1, opt_n4e1, diff_n4e1))

        if out_array_fms is None:
            out_array_fms = new_cols
        else:
            out_array_fms = np.vstack((out_array_fms, new_cols))
    out_array_fms = out_array_fms.astype(np.uint8)
    concated_fms = Image.fromarray(out_array_fms, mode="RGB")

    # concat imgs with fms
    out_imgs_fms = np.hstack((out_array, out_array_fms))
    out_imgs_fms = out_imgs_fms.astype(np.uint8)
    concated_imgs_fms = Image.fromarray(out_imgs_fms, mode="RGB")
    return concated_imgs, concated_fms, concated_imgs_fms, out_not_resize_img


def generate_colormap(featureMap, pixel_max, pixel_min,
                      color_map="nipy_spectral"):
    """Generate colormap.
    """
    featureMap = featureMap.copy()
    cm = plt.get_cmap(color_map)
    featureMap = (featureMap - pixel_min) / (pixel_max - pixel_min)
    colored_featureMap = cm(featureMap)
    return colored_featureMap
