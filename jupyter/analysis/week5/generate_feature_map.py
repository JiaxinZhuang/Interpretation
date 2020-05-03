import os
import sys
import torch
import numpy as np
import pickle
import math

sys.path.append("../../../src/")
sys.path.append("../../")
import model
from utils.function import timethis
from aux.utils import obtain_features_map, zscore


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def fill_AArray_under_resolution(value, width, height):
    aCol = np.repeat(value, height, axis=0)
    array = np.repeat([aCol], width, axis=0)
    return array


def get_rgb_points_by_batch(batch_size=96, width=224, height=224, step=8):
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


def obtain_selected_4D_featureMaps(layer_output_indexes=None,
                                   selected_filter=None,
                                   batch_size=96,
                                   step=8,
                                   net=None, device=None):
    """Args:
        method: [max, median, mean]
    """
    rets_max = []
    rets_mean = []
    rets_median = []

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    dataloader = get_rgb_points_by_batch(batch_size=batch_size, step=step)
    all_iterations = math.ceil((255 / step + 1) ** 3 / batch_size)
    # xs => r, gs => g, zs => b
    xs = []
    ys = []
    zs = []
    for index, (imgs, x, y, z) in enumerate(dataloader):
        if index % 10000 == 0:
            print("[{}/{}]".format(index, all_iterations))
        xs.extend(x)
        ys.extend(y)
        zs.extend(z)
        data = zscore(imgs, mean, std)
        data = torch.tensor(data).to(device)
        layer_output, _ = obtain_features_map(
            data, net.model.features,
            layer_output_indexes=layer_output_indexes)
        ret_max = np.max(layer_output[0][:, selected_filter], axis=(1, 2))
        ret_mean = np.mean(layer_output[0][:, selected_filter], axis=(1, 2))
        ret_median = np.median(layer_output[0][:, selected_filter],
                               axis=(1, 2))
        rets_max.extend(ret_max)
        rets_mean.extend(ret_mean)
        rets_median.extend(ret_median)
        del layer_output
    del data
    rets_max = np.array(rets_max)
    rets_mean = np.array(rets_mean)
    rets_median = np.array(rets_median)
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    zs = np.array(zs, dtype=np.float32)
    return xs, ys, zs, rets_max, rets_mean, rets_median


@timethis
def main():
    resume = "037-0"
    # model_dir = "/home/lincolnzjx/Desktop/Interpretation/saved/models"
    model_dir = "/data/jiaxin/myCode/InterpretationFolder/Interpretation0330/saved/models"
    backbone = "vgg16"
    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    # Hyper parameter.
    step = 1
    batch_size = 330
    layer_output_indexes = [1, 3]
    layer_filter_amount = [64, 64]
    # layer_output_indexes = [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29]
    # layer_filter_amount = [64, 64, 128, 128, 256, 256, 512, 512, 512, 512,
    #                        512, 512, 512]

    for layer_output_index in layer_output_indexes:
        for layer_filter in layer_filter_amount:
            for selected_filter in range(layer_filter):
                print(">> {}-{}".format(layer_output_index,
                                        selected_filter))
                xs, ys, zs, rets_max, rets_mean, rets_median =\
                    obtain_selected_4D_featureMaps(
                        layer_output_indexes=[layer_output_index],
                        selected_filter=selected_filter,
                        batch_size=batch_size,
                        step=step, net=net, device=device)

                save_path = "./RGB_Activation_L{}F{}_{}_{}.pkl"
                for method in ["max", "mean", "median"]:
                    save_path_post = save_path.format(layer_output_index,
                                                      selected_filter,
                                                      method)
                    with open(save_path_post, "wb") as handle:
                        pickle.dump(save_path_post, handle)
                        print(">> Save to {}".format(save_path_post))


if __name__ == "__main__":
    main()
