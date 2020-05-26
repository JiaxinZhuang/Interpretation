"""Visualization for ResNet.
"""
import os
import sys
import torch
from torchvision import transforms
import matplotlib as mpl
from PIL import Image
import argparse


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mpl.rcParams['figure.dpi'] = 600

sys.path.append("../../../src/")
sys.path.append("../../")
import model
from datasets import imagenet
from aux.utils import load_imgs, zscore, extract_valid
from aux.visualization import visualize_features_map_for_comparision


def main(exp, epoch, layer_name, img_index=12):
    save_dir = "/home/lincolnzjx/Desktop/Interpretation/saved/pack/"
    # ################## Hyper-Parameter #######################
    # exp = "033100"
    # epoch = "99900"
    exp = str(exp)
    epoch = str(epoch)
    ##########################################################
    ab_path = os.path.join(save_dir, exp, epoch)

    backbone = "resnet18"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    net = model.Network(backbone=backbone, num_classes=1000)
    net = model.Network(backbone=backbone, num_classes=1000, activations=True,
                        pretrained=True)
    net.to(device)

    # Load data
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = None
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    trainset = imagenet.ImageNet(root="/media/lincolnzjx/HardDisk/Datasets/",
                                 is_train=True, transform=train_transform)

    trainset.set_data([950], 30)
    imgs_path = []
    images = []
    labels = []
    for img, label, img_path in trainset:
        images.append(img.unsqueeze(0))
        labels.append(label)
        imgs_path.append(img_path)

    # Load image
    optimized_data, valid_imgs_path, valid_imgs_index = \
        load_imgs(ab_path, imgs_path, non_exists_ok=True)
    valid_imgs, valid_labels = extract_valid(images, labels, valid_imgs_index)
    optimized_data = zscore(optimized_data, mean, std)

    index2image = {index: item.split("/")[-1].split(".")[0]
                   for index, item in enumerate(valid_imgs_path)}
    index2image

    save_dict = {
        "save_dir": "../../../saved/pack/" + exp + "/feature_map/",
        "index2image": index2image,
        "save_name": "layer-{}-{}-Comparision.pdf"
    }

    try:
        os.mkdir(save_dict["save_dir"])
    except FileExistsError:
        print("Directory has been created {}".format(save_dict["save_dir"]))

    # Move to device
    opt_image = torch.from_numpy(optimized_data).to(device)
    original_image = torch.cat(valid_imgs, dim=0).to(device)

    color_map = "nipy_spectral"
    conv_output_index_dict = {0: 0}
    layer_name = "relu"
    ori_activation_maps = net.get_activation_maps(original_image, layer_name)
    opt_activation_maps = net.get_activation_maps(opt_image, layer_name)

    if img_index != -1:
        visualize_features_map_for_comparision(
            img_index=img_index, layer_index=0,
            features_map=ori_activation_maps,
            opt_feature_map=opt_activation_maps,
            cols=8, conv_output_index_dict=conv_output_index_dict,
            save_dict=save_dict, is_save=True,
            plt_mode="img_scale", color_map=color_map, layer_name=layer_name)
    else:
        for index in range(30):
            visualize_features_map_for_comparision(
                img_index=index, layer_index=0,
                features_map=ori_activation_maps,
                opt_feature_map=opt_activation_maps,
                cols=8, conv_output_index_dict=conv_output_index_dict,
                save_dict=save_dict, is_save=True,
                plt_mode="img_scale", color_map=color_map,
                layer_name=layer_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='viz')
    parser.add_argument("--exp", type=str, default="0514")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--layer", type=str, default='relu')
    parser.add_argument("--epoch", type=int, default=1111)
    parser.add_argument("--img_index", type=int, default=12)
    args = parser.parse_args()
    print(args)

    exp_list = []

    exp_str = args.exp + "{:0>2}"
    start = args.start
    end = args.end
    layer_name = args.layer
    epoch = args.epoch
    img_index = args.img_index

    for index in range(start, end+1):
        exp_list.append(exp_str.format(index))
    epoch_list = [epoch] * len(exp_list)
    for exp, epoch in zip(exp_list, epoch_list):
        main(exp, epoch, layer_name, img_index)
