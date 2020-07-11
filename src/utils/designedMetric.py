"""DesignedMetric.
"""


import numpy as np


def mMetric_v1(ori_activation_maps, opt_activation_maps, selected_filter,
               co1, co2):
    """Metric, the first version.
    Args:
        ori_activation_maps: [batch_size, channels, height, width]
        opt_activation_maps: [batch_size, channels, height, width]
    """
    ori = ori_activation_maps.copy()
    opt = opt_activation_maps.copy()

    ori_sel = ori[selected_filter]
    opt_sel = opt[selected_filter]
    opt_res = np.concatenate((opt[: selected_filter],
                              opt[selected_filter:]), axis=1)
    ori_res = np.concatenate((ori[: selected_filter],
                              ori[selected_filter:]), axis=1)

    batch_size, channels, height, width = ori.size()
    pixels_per_image = channels * height * width

    # SF term.
    SF_0 = np.sum(opt_sel == 0, axis=(1, 2, 3))
    SF_term = SF_0 / pixels_per_image
    print("=> SF_0: {}".format(SF_0))
    print("=> pixels_per_image: {}".format(pixels_per_image))
    print("=> SF_term: {}".format(SF_term))

    # RF term.
    ori_sel = np.expand_dims(ori_sel, axis=1)
    ori_sel = np.repeat(ori_sel, channels-1, axis=1)
    ori_sel_mask = ori_sel == 0
    opt_res = opt_res * ori_sel_mask
    pixel_per_image = np.sum(ori_sel_mask, axis=(1, 2, 3))
    RF_0_non_overlap = np.sum(opt_res[ori_res == 0], axis=(1, 2, 3))
    RF_term = RF_0_non_overlap / pixel_per_image

    over_metric = (co1 * SF_term + co2 * RF_term) / (co1 + co2)
    return over_metric
