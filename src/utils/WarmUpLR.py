"""Warmup.
"""

from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    """Warmup_training learning rate schduler.

    Args:
        optimizer: optimizer(e.g. SGD)
        total_iters: total_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Use the first m batches, and set the learning rate to
        base_lr * m / total_iters.
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8)
                for base_lr in self.base_lrs]
