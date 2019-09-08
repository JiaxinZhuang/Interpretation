import os
import logging
import sys


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

from tqdm import tqdm
import numpy as np
import torch

sys.path.append("./src/utils")


def init_environment(seed=0, cuda_id=0):
    """Init environment
    initialize environment including cuda, benchmark, random seed, saved model
    directory and saved logs directory
    """
    print(">< init_environment with seed: {}".format(seed))

    cuda_id = str(cuda_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_id


    if seed != -1:
        print("> Use seed -{}".format(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        print("> Don't use seed")


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