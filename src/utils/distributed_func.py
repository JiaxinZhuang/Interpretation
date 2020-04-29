"""Function for auxilary distributed training.
"""

import torch.distributed as dist


def reduce_tensor(tensor, world_size=1):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt
