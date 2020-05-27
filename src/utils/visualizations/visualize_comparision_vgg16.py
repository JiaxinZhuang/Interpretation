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

import model
from datasets import imagenet
sys.path.append("jupyter/")
from aux.utils import obtain_features_map, load_imgs, zscore,\
    extract_valid
from aux.visualization import visualize_features_map_for_comparision


def main(exp, epoch, layer_index, img_index=12,
         class_index=950, num_classes=30, data_dir=None):
    save_dir = "./saved/generated/"
    # ################## Hyper-Parameter #######################
    # exp = "033100"
    # epoch = "99900"
    exp = str(exp)
    epoch = str(epoch)
    ##########################################################
    ab_path = os.path.join(save_dir, exp, epoch)

    resume = "037-0"
    model_dir = "../../../saved/models"
    backbone = "vgg16"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    net = model.Network(backbone=backbone, num_classes=1000)
    net.to(device)
    net.eval()

    # resume from model
    resume_exp = resume.split("-")[0]
    resume_epoch = resume.split("-")[1]
    print("Resume from model from exp: {} at epoch {}".format(resume_exp,
                                                              resume_epoch))
    resume_path = os.path.join(model_dir, str(resume_exp), str(resume_epoch))
    ckpt = torch.load(resume_path, map_location=device)
    net.load_state_dict(ckpt)

    # Load data
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = None
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    trainset = imagenet.ImageNet(root=data_dir, is_train=True,
                                 transform=train_transform)

    trainset.set_data([class_index], num_classes)

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
        "save_dir": "./saved/generated/" + exp + "/feature_map/",
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

    # hyper parameter
    conv_output_indexes = [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29]
    conv_output_indexes_dict = dict(zip(conv_output_indexes,
                                        range(len(conv_output_indexes))))
    print(conv_output_indexes_dict)

    conv_output, _ = \
        obtain_features_map(original_image, net.model.features,
                            layer_output_indexes=conv_output_indexes)
    opt_feature_map, _ = \
        obtain_features_map(opt_image, net.model.features,
                            layer_output_indexes=conv_output_indexes)

    if img_index != -1:
        visualize_features_map_for_comparision(
            img_index=img_index, layer_index=layer_index,
            features_map=conv_output,
            opt_feature_map=opt_feature_map, cols=4,
            conv_output_index_dict=conv_output_indexes_dict,
            save_dict=save_dict,
            plt_mode="img_scale", color_map="nipy_spectral")
    else:
        for index in range(30):
            visualize_features_map_for_comparision(
                img_index=index, layer_index=layer_index,
                features_map=conv_output,
                opt_feature_map=opt_feature_map, cols=4,
                conv_output_index_dict=conv_output_indexes_dict,
                save_dict=save_dict,
                plt_mode="img_scale", color_map="nipy_spectral")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='viz')
    parser.add_argument("--exp", type=str, default="0514")
    parser.add_argument("--layer", type=str, default='relu')
    parser.add_argument("--epoch", type=int, default=1111)
    parser.add_argument("--img_index", type=int, default=12)
    parser.add_argument("--num_classes", type=int, default=30)
    parser.add_argument("--class_index", type=int, default=950)
    parser.add_argument("--server", default="local", type=str,
                        choices=["local", "ls15", "ls16", "ls31",
                                 "ls97", "desktop"],
                        help="server to run the code")
    args = parser.parse_args()
    print(args)

    main(args.exp, args.epoch, args.layer, args.img_index,
         args.class_index, args.num_classes, args.server)
