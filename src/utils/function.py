import os
import logging
import sys
import copy
from functools import wraps
import time
from datetime import timedelta
import gc

import resource

import numpy as np
import torch
from PIL import Image


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


sys.path.append("./src/utils")


def init_environment(seed=0, cuda_id=0, _print=None):
    """Init environment
    initialize environment including cuda, benchmark, random seed, saved model
    directory and saved logs directory
    """
    _print(">< init_environment with seed: {}".format(seed))

    cuda_id = str(cuda_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_id
    _print("Using GPU ID {}".format(cuda_id))

    if seed != -1:
        _print("> Use seed -{}".format(seed))
        # Determnistic mode would get determinitic algorithm for conv
        # Ref: https://zhuanlan.zhihu.com/p/39752167
        torch.backends.cudnn.deterministic = True
        # benchmark mode would look for the optimial set of algorithms for the
        # parriclar configuaration, which usually leads to faster runtime.
        # Input size should not vary every iteration, and conputation graph is
        # unchange.
        # Ref: https://discuss.pytorch.org/t/
        # what-does-torch-backends-cudnn-benchmark-do/5936/2
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def init_logging(output_dir, exp):
    """Init logging and return logging
        init_logging should used after init environment
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(message)s",
                        datefmt="%Y%m%d-%H:%M:%S",
                        filename=os.path.join(output_dir, str(exp) + ".log"),
                        filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)
    return logging


def timethis(func, *args, **kwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        elapse = time.time() - start_time
        elapse = str(timedelta(seconds=elapse))
        print(">> Functoin: {} costs {}".format(func.__name__, elapse))
        sys.stdout.flush()
        return ret
    return wrapper


def str2bool(val):
    """convert str to bool
    """
    value = None
    if val == 'True':
        value = True
    elif val == 'False':
        value = False
    else:
        raise ValueError
    return value


def str2list(val):
    """convert str to bool
    """
    value = val.split(",")
    value = [int(v.strip()) for v in value]
    return value


def preprocess_image(pil_im, mean, std, resize=512, resize_im=True,
                     device=None):
    """Process images for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    Returns:
        im_as_var (torch variable): Variable that contains processed
                                    float tensor
    """
    # Resize image
    if resize_im:
        pil_im.resize((resize, resize))

    im_as_arr = np.float32(pil_im)
    if len(im_as_arr.shape) == 2:
        im_as_arr = np.expand_dims(im_as_arr, axis=2)
    # Convert array to [D, W, H]
    im_as_arr = im_as_arr.transpose(2, 0, 1)
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).to(torch.float32)
    # Add one more channle to the beginning.
    im_as_ten.unsqueeze_(0)
    im_as_var = im_as_ten.clone().detach().to(device).requires_grad_(True)
    return im_as_var


def format_np_output(np_arr):
    """This is a (kind of) bandiad fix to steamline save procedure.
       It converts all the outputs to the same format which
       is 3xWxH with using successive if clauses.

    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH.
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase / Case 4: Np arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)

    if len(np_arr.shape) == 3 and np_arr.shape[2] == 1:
        np_arr = np_arr.squeeze(axis=2)
    return np_arr


def save_image(im, path):
    """Save a numpy matrix or PIL image as an image.

    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def save_numpy(arr, path):
    """Save a numpy matrix.

    Args:
        arr (Numpy array): Matrix of shape BXDxWxH
        path (str): Path to the image
    """
    try:
        if isinstance(arr, (np.ndarray, np.generic)):
            # with open(path, "w") as handle:
            np.save(path, arr)
    except FileNotFoundError:
        print("Can't save arr to {}".format(path))
        sys.exit(-1)


def recreate_image(im_as_var, reverse_mean, reverse_std, rescale=False):
    """Recreate images from a torch variable, sort of reverse preprocessing.

    Args:
        im_as_var (torch variable): Image to recreate
    Returns:
        recreate_im (numpy arr): Recreated image in array
    """
    recreate_im = copy.copy(im_as_var)
    assert len(recreate_im.shape) == 3
    channels = recreate_im.shape[0]
    for channel in range(channels):
        recreate_im[channel] /= reverse_std[channel]
        recreate_im[channel] -= reverse_mean[channel]

    if rescale:
        recreate_im = _rescale(recreate_im)
    else:
        recreate_im[recreate_im > 1] = 1
        recreate_im[recreate_im < 0] = 0

    recreate_im = np.round(recreate_im * 255)

    recreate_im = np.uint8(recreate_im).transpose(1, 2, 0)
    return recreate_im


def _rescale(recreate_im):
    """Rescale images to 0-1 from a more wide range.
    Args:
        recreate_im: [channels, height, width]
    Returns:
        recreate_im: [channels, height, width]
    """
    recreate_im = recreate_im - np.amin(recreate_im)
    recreate_im = recreate_im / np.amax(recreate_im)
    return recreate_im


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_grad_norm(processed_images):
    """Get grad norm L2 for processed_images.
    """
    grads = processed_images.grad.clone().cpu()
    grads_norm = torch.norm(grads)
    return grads_norm


# def get_net_grad_norm(net):
#     """Get grad norm L2 for net."""
#     grad_norm =
#     for name, param in net.named_parameters():


def dataname_2_save(imgs_path, saved_dir, epoch=None):
    """Img path saved.
    """
    output_name = []
    for name in imgs_path:
        name = name.split("/")[-1:]
        output = os.path.join(saved_dir, *name)
        if epoch:
            output = os.path.splitext(output)[0] + "_" + str(epoch) + ".png"
        else:
            output = os.path.splitext(output)[0] + ".png"
        output_name.append(output)
    # print(output_name)
    return output_name


def _test_image_related():
    """Test preprocess_image and save_image.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    image = Image.open("../../data/example/dd_tree.jpg").convert("RGB")
    im_as_var = preprocess_image(image, mean=mean, std=std)
    print(im_as_var.size())
    recreate_im = recreate_image(im_as_var, reverse_mean, reverse_std)
    print(recreate_im.shape)
    save_image(recreate_im, "../../saved/generated/recreate_im.jpg")


def _test_dataname2save():
    """Test dataname_2_save.
    """
    saved_dir = "/media/lincolnzjx/Disk21/interpretation/saved/generated"
    imgs_path = ["../data/CUB_200_2011/images/001.Black_footed_Albatross\
                 /Black_Footed_Albatross_0009_34.jpg",
                 "../data/CUB_200_2011/images/034.Gray_crowned_Rosy_Finch\
                 /Gray_Crowned_Rosy_Finch_0044_26976.jpg"]
    output_name = dataname_2_save(imgs_path, saved_dir)
    print("Test dataname2save")
    print(imgs_path)
    print("-"*80)
    print(output_name)


def freeze_model(net, _print=None):
    """Freeze model.
    """
    for param in net.parameters():
        param.requires_grad = False

    for name, param in net.named_parameters():
        _print("Name:{}, Required Grad: {}".format(name, param.requires_grad))
    return net


def format_print(array, name_list):
    str_format = "{}:{:.3f}\t"
    output_str = ""
    base = array[0]
    if base == 0:
        print("Base can't be zero")
        sys.exit(-1)
    for item, name in zip(array, name_list):
        output = item / base
        output_str += str_format.format(name, output)
    print(output_str)


def adjust_learning_rate(init_lr, optimizer, epoch, step, len_epoch):
    """Lr sechdule that should yield 76% converged accuracy with batch size
    256 for ImageNet.
    ???
    """
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = init_lr * (0.1 ** factor)

    # Warm up
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def to_python_float(data):
    """Convert torch float to python float"""
    if hasattr(data, "item"):
        return data.item()
    else:
        return data[0]


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_embeddings(loader, net, save_memory=False):
    """Get embeddings according to net given loader, train or test dataset
    """
    def _get_raw_pixel(loader):
        embeddings_cpu = []
        targets_cpu = []
        for _, (data, targets) in enumerate(loader):
            data = data.view(data.size(0), -1)
            targets = targets.cpu()
            embeddings_cpu.append(data)
            targets_cpu.append(targets)
        embeddings_cpu = torch.cat(embeddings_cpu, dim=0)
        targets_cpu = torch.cat(targets_cpu, dim=0)
        return embeddings_cpu, targets_cpu

    if net is None:
        return _get_raw_pixel(loader)

    net.eval()
    embeddings_cpu = []
    targets_cpu = []
    with torch.no_grad():
        for _, (data, targets) in enumerate(loader):
            data = data.cuda()
            output = net.get_embeddings(data)
            output = output.cpu()
            targets = targets.cpu()
            embeddings_cpu.append(output)
            targets_cpu.append(targets)
    embeddings_cpu = torch.cat(embeddings_cpu, dim=0)
    targets_cpu = torch.cat(targets_cpu, dim=0)
    return embeddings_cpu, targets_cpu


def zscore(optimized_data, mean, std):
    """Zscore the data.
    Args:
        optimized_data: [batch_size, height, width, channels]
        mean: [channels]
        std: [channels]
    Return:
        optimized_data: [batch_size, channels, heights, width]
    """
    print(optimized_data.shape)
    if optimized_data.shape[1] != 3 and optimized_data.shape[1] != 1:
        optimized_data = np.transpose(optimized_data.copy(), (0, 3, 1, 2))
    else:
        optimized_data = optimized_data.copy()
    channels = optimized_data.shape[1]
    for channel in range(channels):
        optimized_data[:, channel, :, :] = optimized_data[:, channel, :, :] -\
            mean[channel]
        optimized_data[:, channel, :, :] = optimized_data[:, channel, :, :] /\
            std[channel]
    return optimized_data


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


if __name__ == "__main__":
    # _test_dataname2save()
    format_print((1, 1, 2), ["1", "2", "3"])
