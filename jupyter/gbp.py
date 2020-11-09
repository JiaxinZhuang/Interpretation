import os
import torch
import numpy as np
from torchvision import transforms
import matplotlib as mpl
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy
import sys
import warnings
# from numba import jit
from functools import wraps
import gc
# import time
# from pympler import tracker, summary, muppy
# from multiprocessing import Pool
from torch.multiprocessing import Pool, Process, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass


warnings.filterwarnings("ignore")

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mpl.rcParams['figure.dpi'] = 600

sys.path.append("/home/lincolnzjx/Desktop/Interpretation/extra/" +
                "pytorch-cnn-visualizations/src/")
sys.path.append("/home/lincolnzjx/Desktop/Interpretation/src/")
sys.path.append("/home/lincolnzjx/Desktop/Interpretation/")

from utils.visualizations.visualize import visualize
import model
from datasets import imagenet
from utils.function import recreate_image
from layer_activation_with_guided_backprop import GuidedBackprop


device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def show_memory(unit='MB', threshold=1):
#     '''查看变量占用内存情况
#
#     :param unit: 显示的单位，可为`B`,`KB`,`MB`,`GB`
#     :param threshold: 仅显示内存数值大于等于threshold的变量
#     '''
#     from sys import getsizeof
#     scale = {'B': 1, 'KB': 1024, 'MB': 1048576, 'GB': 1073741824}[unit]
#     for i in list(globals().keys()):
#         memory = eval("getsizeof({})".format(i)) // scale
#         if memory >= threshold:
#             print(i, memory)


# def timethis(func, *args, **kwargs):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         ret = func(*args, **kwargs)
#         elapse = time.time() - start_time
#         print(">> Functoin: {} costs {:.4f}s".format(func.__name__, elapse))
#         sys.stdout.flush()
#         return ret
#     return wrapper


def gcollect(func, *args, **kwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        gc.collect()
        # print(">> Function: {} has been garbage collected".
        #       format(func.__name__))
        sys.stdout.flush()
        return ret
    return wrapper


@gcollect
def gbp(prep_img, pretrained_model, selected_layer, selected_filter,
        device=None):
    GBP = GuidedBackprop(pretrained_model)
    prep_img = torch.tensor(prep_img, device=device, requires_grad=True)
    guided_grads = GBP.generate_gradients(prep_img, None, selected_layer,
                                          selected_filter)
    del GBP, prep_img
    return guided_grads


@gcollect
def main(selected_layer=None, selected_filter=None, class_index=None,
         img=None, img_path=None, net=None, img_index=None, num_class=100):

    print(selected_layer, selected_filter, img_path)
    # dirs = "/home/lincolnzjx/Desktop/Interpretation/saved/generated/GBP/"
    dirs = "/media/lincolnzjx/HardDisk/myGithub/Interpretation/generated/GBP/"
    dirs = os.path.join(dirs, str(class_index))

    dir_name = "opt"
    dir_name_fm = "fm"

    dir_path = os.path.join(dirs, dir_name)
    dir_path_fm = os.path.join(dirs, dir_name_fm)

    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(dir_path_fm, exist_ok=True)

    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

    name = img_path.split("/")[-1].split(".")[0]
    img = img[0].permute((1, 2, 0)).numpy()
    X = (torch.FloatTensor(img[numpy.newaxis].transpose([0, 3, 1, 2])*1) -
         mean) / std

    # Ori
    X = X.to(device)
    ori_activation_maps = net.get_activation_maps(X, selected_layer)
    assert len(ori_activation_maps) == 1
    assert len(ori_activation_maps[0]) == 1
    sel_feature_map = ori_activation_maps[0][0, selected_filter]
    min_val = sel_feature_map.min()
    max_val = sel_feature_map.max()
    sel_feature_map -= sel_feature_map.min()
    sel_feature_map /= sel_feature_map.max()
    cm = plt.get_cmap("jet")
    sel_feature_map = cm(sel_feature_map)
    sel_feature_map = (sel_feature_map[:, :, :3] * 255).astype(np.uint8)

    # GBP
    pretrained_model = torchvision.models.vgg16(pretrained=True).to(device)
    guided_grads = gbp(X, pretrained_model,
                       selected_layer, selected_filter)

    name = ("layer_{}_filter_{}_".format(selected_layer,
                                         selected_filter)) + name
    path = os.path.join(dir_path, name+".png")

    dir_path_fm_sub = os.path.join(dir_path_fm, name)
    os.makedirs(dir_path_fm_sub, exist_ok=True)
    fm_name = "feature_map_filter_{}".format(selected_filter)
    fm_path = os.path.join(dir_path_fm_sub, fm_name + ".png")

    # Standalize
    # relevance_img = standalize(R[0][0], X.numpy(), path)
    # fm = visualize_sel_fm(net, relevance_img.transpose(2,0,1),
    #                       selected_layer, selected_filter, min_val,
    #                       max_val)
    # fm = (255 * fm).astype(np.uint8)
    # Image.fromarray(fm).save(fm_path)

    # Standlize remove negative
    # relevance_img = standalize_remove_negative(R[0][0], X.numpy(), path)
    # fm = visualize_sel_fm(net, relevance_img.transpose(2,0,1),
    #                       selected_layer, selected_filter, min_val,
    #                       max_val)
    # fm = (255 * fm).astype(np.uint8)
    # Image.fromarray(fm).save(fm_path)

    # Scale positive
    # relevance_img = scale_positive(R[0][0], X.numpy(), path)
    # fm = visualize_sel_fm(net, relevance_img.transpose(2,0,1),
    #                       selected_layer, selected_filter, min_val,
    #                       max_val)
    # fm = (255 * fm).astype(np.uint8)
    # Image.fromarray(fm).save(fm_path)

    # Scale total
    relevance_img = scale_total(guided_grads, X.cpu().numpy()[0], path)
    fm = visualize_sel_fm(net, relevance_img, selected_layer,
                          selected_filter, min_val, max_val)
    fm = (255 * fm).astype(np.uint8)

    plt.savefig(fm_path)

    img = np.array(Image.open(path)).astype(np.float64) / 255.0
    X = torch.from_numpy((img[numpy.newaxis]).transpose([0, 3, 1, 2]))
    X = (X-mean) / std
    X = X.to(device).float()
    opt_activation_maps = net.get_activation_maps(X,
                                                  selected_layer)
    visualize(ori_activation_maps, opt_activation_maps,
              img_index=img_index, layer_name=selected_layer,
              backbone="vgg16", num_class=num_class,
              exp="GBP/{}".format(class_index),
              imgs_path=[dir_path_fm_sub])
    del opt_activation_maps, ori_activation_maps
    del X
    del net
    del pretrained_model
    del guided_grads
    del relevance_img
    del fm
    del img


@gcollect
def visualize_sel_fm(net, relevance_img, selected_layer, selected_filter,
                     min_val, max_val):
    cm = plt.get_cmap("jet")
    opt_activation_maps = net.get_activation_maps(torch.FloatTensor(
        relevance_img).unsqueeze(dim=0).to(device), selected_layer)
    opt_sel_feature_map = opt_activation_maps[0][0, selected_filter]
    opt_sel_feature_map -= min_val
    opt_sel_feature_map = np.maximum(0, opt_sel_feature_map)
    opt_sel_feature_map /= (opt_sel_feature_map.max() + 1e-8)
    opt_sel_feature_map = np.minimum(1, opt_sel_feature_map)
    opt_sel_feature_map = cm(opt_sel_feature_map)
    del opt_activation_maps
    return opt_sel_feature_map


@gcollect
def scale_total(relevance, normalized, path=None):
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]

    relevance = np.array(relevance)
    relevance /= np.max(np.abs(relevance))
    relevance_img = normalized * relevance
    new_image = recreate_image(relevance_img, reverse_mean, reverse_std)
    Image.fromarray(new_image).save(path)
    del relevance, normalized, new_image
    return relevance_img


def trainer():
    csv_file = pd.read_excel("/home/lincolnzjx/Desktop/Interpretation/jupyter/"
                             + "filter.xlsx", header=None)
    csv_file = csv_file.values
    num_class = 100
    backbone = "vgg16"

    # dicts = {
    #      948: [1, 3],
    #      444: [13, 20],
    #      522: [14, 21],
    #      14: [25, 3],
    #      84: [29, 11]
    # }
    class_index = None

    con_run_process = 10
    '''
    for item1, item2, _ in csv_file:
        if isinstance(item1, str) and item1.startswith("class"):
            class_index = int(item1.replace("class", ""))
            continue

        # ONLY RUN 950 -------
        if class_index != 950:
            continue
        # ONLY RUN 522 -------

        selected_filter = int(item1)
        selected_layer = int(item2)
        # Model
        net = model.Network(backbone=backbone, num_classes=1000,
                            selected_layer=selected_layer)
        # resume from model
        resume = "037-0"
        model_dir = "/home/lincolnzjx/Desktop/Interpretation/saved/models"
        resume_exp = resume.split("-")[0]
        resume_epoch = resume.split("-")[1]
        resume_path = os.path.join(model_dir, str(resume_exp),
                                   str(resume_epoch))
        ckpt = torch.load(resume_path, map_location=device)
        net.load_state_dict(ckpt, strict=False)
        net.to(device)
        net.eval()

        # Datasets.
        train_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.ToTensor()
        ])
        trainset = imagenet.ImageNet(root="/media/lincolnzjx/HardDisk/" +
                                     "Datasets/",
                                     is_train=True,
                                     transform=train_transform)
        trainset.set_data([class_index], num_class)

        processes = Pool(processes=con_run_process)
        for index, (img, label, img_path) in enumerate(trainset):
            img = img.unsqueeze(0)
            processes.apply_async(main, args=(selected_layer,
                                              selected_filter,
                                              class_index,
                                              img, img_path,
                                              net, index))
        processes.close()
        processes.join()
    '''

    selected = [
        [1, 53],
        [1, 57],
        [1, 60],
        [1, 61],
        [3, 56],
        [3, 60],
        [3, 28],
        [3, 19],
        [6, 19],
        [6, 117]
    ]
    '''
        [1, 47],
        [1, 16],
        [3, 20],
        [3, 41],
        [6, 114],
        [6, 76],
        [8, 17],
        [8, 99],
        [11, 174],
        [11, 75],
        [13, 98],
        [13, 21],
        [15, 102],
        [15, 173],
        [18, 458],
        [18, 353],
        [20, 46],
        [20, 103],
        [22, 214],
        [22, 164],
        [25, 19],
        [25, 1],
        [27, 161],
        [27, 476],
        [29, 64],
        [29, 101]
    ]
    '''
    class_indexes = [950]

    for class_index in class_indexes:
        for selected_layer, selected_filter in selected:
            # Model
            net = model.Network(backbone=backbone, num_classes=1000,
                                selected_layer=selected_layer)
            # resume from model
            resume = "037-0"
            model_dir = "/home/lincolnzjx/Desktop/Interpretation/saved/models"
            resume_exp = resume.split("-")[0]
            resume_epoch = resume.split("-")[1]
            resume_path = os.path.join(model_dir, str(resume_exp),
                                       str(resume_epoch))
            ckpt = torch.load(resume_path, map_location=device)
            net.load_state_dict(ckpt, strict=False)
            net.to(device)
            net.eval()

            # Datasets.
            train_transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=Image.BILINEAR),
                transforms.ToTensor()
            ])
            trainset = imagenet.ImageNet(root="/media/lincolnzjx/HardDisk/" +
                                         "Datasets/",
                                         is_train=True,
                                         transform=train_transform)
            trainset.set_data([class_index], num_class)

            processes = Pool(con_run_process)
            for index, (img, label, img_path) in enumerate(trainset):
                if index != 12:
                    continue
                img = img.unsqueeze(0)
                processes.apply_async(main, args=(selected_layer,
                                                  selected_filter,
                                                  class_index,
                                                  img, img_path,
                                                  net, index))
            processes.close()
            processes.join()


if __name__ == "__main__":
    trainer()
