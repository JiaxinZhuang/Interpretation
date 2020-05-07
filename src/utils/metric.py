"""Metric
"""

import numpy as np
import torch


def mean_class_recall(y_true, y_pred):
    """Mean class recall.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    class_recall = []
    target_uniq = np.unique(y_true)

    for label in target_uniq:
        indexes = np.nonzero(label == y_true)[0]
        recall = np.sum(y_true[indexes] == y_pred[indexes]) / len(indexes)
        class_recall.append(recall)
    return np.mean(class_recall)


def accuracy(output, target, topk=(1,)):
    """Compute the precision@k for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        # [maxk, batch_size]
        pred = pred.t()
        # [1, batch_size] -> [maxk, batch_size]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# class AverageMeter():
#     """Compute and stores the average and current value.
#     """
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0.0
#         self.avg = 0.0
#         self.sum = 0.0
#         self.count = 0.0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    # y_pred = [0, 0, 1, 1, 2, 2, 2]
    # y_true = [0, 0, 0, 0, 1, 2, 2]
    # print(mean_class_recall(y_true, y_pred))
    # top1 = AverageMeter()
    # top1.update(10, 10)
    # print(top1.avg)
    # top1.update(20, 10)
    # print(top1.avg)
    # top1.update(30, 20)
    # print(top1.avg)
    output = torch.randn(3, 6)
    target = torch.tensor([1, 4, 2])
    print(output)
    print(target)
    (acc1, acc5) = accuracy(output, target, (1, 5))
    print(acc1, acc5)
    print(acc1.item(), acc5.item())
