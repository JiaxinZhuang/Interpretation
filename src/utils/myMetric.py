import torch
import numpy as np
import sys
sys.path.append("../")
import pytorch_ssim


def mMetric_v3(ori_activation_maps, opt_activation_maps, selected_filter,
               co1=1, co2=1):
    ori = ori_activation_maps.copy()
    opt = opt_activation_maps.copy()
    # diff = ori_activation_maps-opt_activation_maps

    # diff_sel = diff[:, selected_filter]
    ori_sel = ori[:, selected_filter]
    opt_sel = opt[:, selected_filter]
    opt_res = np.concatenate((opt[:, : selected_filter],
                              opt[:, selected_filter+1:]), axis=1)
    # ori_res = np.concatenate((ori[:, : selected_filter],
    #                           ori[:, selected_filter+1:]), axis=1)

    (batch_size, channels, height, width) = ori.shape

    ori_sel_ts = torch.from_numpy(ori_sel).unsqueeze(dim=1).cuda()
    opt_sel_ts = torch.from_numpy(opt_sel).unsqueeze(dim=1).cuda()

    ssim_sel = []
    rmses = []
    for index in range(30):
        ssim = pytorch_ssim.ssim(ori_sel_ts[index].unsqueeze(dim=0),
                                 opt_sel_ts[index].unsqueeze(dim=0)).cpu().\
            numpy()
        rmse = torch.sqrt(((ori_sel_ts[index].unsqueeze(dim=0) -
                            opt_sel_ts[index].unsqueeze(dim=0)) ** 2).mean()).\
            cpu().numpy()
        ssim_sel.append(ssim)
        rmses.append(rmse)
        # ssim_loss = pytorch_ssim.SSIM(window_size=11)
        # print(ssim_loss(ori_sel, opt_sel))

    # RF
    ori_sel = np.expand_dims(ori_sel, axis=1)
    # print("ori_sel: ", ori_sel.shape)
    ori_sel = np.repeat(ori_sel, channels-1, axis=1)
    # print("ori_sel: ", ori_sel.shape)
    ori_sel_mask = ori_sel == 0
    opt_res_mask = opt_res != 0
    new_opt_res = opt_res * ori_sel_mask * opt_res_mask
    opt_res_ts = torch.from_numpy(new_opt_res).cuda()
    # print(opt_res_ts.shape)
    zero_mask = torch.zeros_like(opt_res_ts).cuda()

    ssim_res = []
    for index in range(30):
        ssim = pytorch_ssim.ssim(opt_res_ts[index].unsqueeze(dim=0),
                                 zero_mask[index].unsqueeze(dim=0)).cpu().\
            numpy()
        ssim_res.append(ssim)
    print("RMSE: {} - std {}".format(np.mean(rmses), np.std(rmses)))
    print("SSIM_SEL: {} - std {}".format(np.mean(ssim_sel), np.std(ssim_sel)))
    print("SSIM_RES: {} - std {}".format(np.mean(ssim_res), np.std(ssim_res)))
    lamba1 = 0.5
    lamba2 = 0.5
    ssim_sum = lamba1 * np.array(ssim_sel) + lamba2 * np.array(ssim_res)
    print("SSIM_SUM {} - std {}".format(np.mean(ssim_sum), np.std(ssim_sum)))
