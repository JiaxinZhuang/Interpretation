"""Visualize.
"""

# import visualize_comparision_resNet18
# import visualize_comparision_vgg16
import sys
import os
sys.path.append("jupyter/")
from aux.visualization import visualize_features_map_for_comparision


def visualize(ori_activation_maps, opt_activation_maps,
              img_index=-1, layer_name="relu", backbone=None, num_classes=30,
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
        for index in range(num_classes):
            visualize_features_map_for_comparision(
                img_index=index, layer_index=layer_index,
                features_map=ori_activation_maps,
                opt_feature_map=opt_activation_maps,
                cols=8, conv_output_index_dict=conv_output_index_dict,
                save_dict=save_dict, is_save=True,
                plt_mode="img_scale", color_map=color_map,
                layer_name=layer_name)
