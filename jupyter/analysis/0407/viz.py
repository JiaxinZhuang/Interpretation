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
from aux.utils import obtain_features_map, load_imgs, zscore, extract_valid
from aux.visualization import visualize_features_map_for_comparision


def main(exp, epoch, layer_index, img_index=12):
    save_dir = "/home/lincolnzjx/Desktop/Interpretation/saved/pack/"
    # ################## Hyper-Parameter #######################
    # exp = "033100"
    # epoch = "99900"
    exp = str(exp)
    epoch = str(epoch)
    ##########################################################
    ab_path = os.path.join(save_dir, exp, epoch)

    resume = "037-0"
    model_dir = "../../../saved/models"
    # generated_dir = "../../../saved/pack/"
    backbone = "vgg16"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    net = model.Network(backbone=backbone, num_classes=1000)
    net.to(device)

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
        # transforms.RandomCrop(input_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    trainset = imagenet.ImageNet(root="/media/lincolnzjx/HardDisk/Datasets/",
                                 is_train=True, transform=train_transform)

    trainset.set_data([950], 30)
    # image, label, imgs_path = trainset
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
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=1)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=1111)
    parser.add_argument("--img_index", type=int, default=12)
    args = parser.parse_args()
    print(args)

    exp_list = []

    exp_str = args.exp + "{:0>2}"
    start = args.start
    end = args.end
    layer_index = args.layer
    epoch = args.epoch
    img_index = args.img_index

    for index in range(start, end+1):
        exp_list.append(exp_str.format(index))
    epoch_list = [epoch] * len(exp_list)
    for exp, epoch in zip(exp_list, epoch_list):
        main(exp, epoch, layer_index, img_index)

    # Layer-6-Filter-19
    # exp_str = "0401{:0>2}"
    # start = 0, end = 31

    # Layer-1-Filter-47
    # exp_str = "0330{:0>2}"
    # start = 4
    # end = 19
    # layer_index = 1
    # exp_str = "0514{:0>2}"
    # start = 57
    # end = 57
    # layer_index = 1

    # Layer-3-Filter-20
    # exp_str = "0331{:0>2}"
    # start = 0
    # end = 18
    # layer_index = 3

    # Layer-6-Filter-19
    # layer_index = 6
    # exp_str = "0401{:0>2}"
    # start = 0
    # end = 19
    # exp_str = "0406{:0>2}"
    # start = 0
    # end = 8
    # exp_str = "0410{:0>2}"
    # start = 10
    # end = 17
    # exp_str = "0411{:0>2}"
    # start = 30
    # end = 38

    # Layer-13-Filter-112
    # layer_index = 13
    # exp_str = "0402{:0>2}"
    # start = 30
    # end = 41
    # exp_str = "0512{:0>2}"
    # start = 27
    # end = 27
    # start = 24
    # end = 24
