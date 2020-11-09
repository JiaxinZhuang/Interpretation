import torch
import numpy as np
from PIL import Image
import sys
import csv
sys.path.append('/home/lincolnzjx/Desktop/Interpretation/src/utils/')
from function import gcollect

device = torch.device('cpu')
@gcollect
def main(selected_layer, selected_filter, net, ori_path, opt_path,
         writer, writer_path):
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

    # ori
    ori_img = np.array(Image.open(ori_path).resize((224, 224))).astype(np.float64) / 255.0
    ori_img = torch.from_numpy((ori_img[np.newaxis]).transpose([0, 3, 1, 2]))
    ori_img = (ori_img-mean) / std
    ori_img = ori_img.float().to(device)
    ori_activation_maps = net.get_activation_maps(ori_img,
                                                  selected_layer)[0]
    # opt
    opt_img = np.array(Image.open(opt_path).resize((224, 224))).astype(np.float64) / 255.0
    opt_img = torch.from_numpy((opt_img[np.newaxis]).transpose([0, 3, 1, 2]))
    opt_img = (opt_img-mean) / std
    opt_img = opt_img.float().to(device)
    opt_activation_maps = net.get_activation_maps(opt_img,
                                                  selected_layer)[0]
    sel_error = np.abs((ori_activation_maps[0, selected_filter] -
                        opt_activation_maps[0, selected_filter])).mean()
    res_error = np.abs(opt_activation_maps[0])
    res_error = res_error.reshape((res_error.shape[0], -1))
    res_error = np.mean(res_error, axis=1)
    res_error = np.delete(res_error, selected_filter, axis=0)
    del opt_activation_maps, ori_activation_maps
    del ori_img, opt_img
    writer(sel_error, res_error, writer_path)
    return sel_error, res_error


def writer(sel_error, res_error, filename):
    with open(filename, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([sel_error])
        writer.writerow(res_error)
