# copyright 2019 jiaxin zhuang
#
#
# ?? license
# ==============================================================================
"""Baseline.

Baseline model

"""
import sys
import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

from datasets import imagenet_dali
from model import init_weights
from utils.function import init_logging, init_environment, get_lr, \
    timethis, adjust_learning_rate, to_python_float, ProgressMeter
# freeze_model,
from utils.metric import AverageMeter, accuracy
# from utils.distributed_func import reduce_tensor
import config
import model


def train(train_loader, net, criterion, optimizer, epoch,
          batch_size=256, prof=False, distributed=False,
          world_size=1, init_lr=1e-2):
    """Train epoch."""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    train_loader_len = int(math.ceil(train_loader._size / batch_size))
    progress = ProgressMeter(
                train_loader_len,
                [batch_time, data_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch)
    )

    # Switch to train mode.
    net.train()
    end = time.time()
    for index, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        imgs = data[0]["data"].cuda(non_blocking=True)
        target = data[0]["label"].squeeze().long().cuda(non_blocking=True)

        adjust_learning_rate(init_lr, optimizer, epoch, index,
                             train_loader_len)

        if prof and index > 10:
            break

        output = net(imgs)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # if distributed:
        #     reduced_loss = reduce_tensor(loss)
        #     acc1 = reduce_tensor(acc1)
        #     acc5 = reduce_tensor(acc5)
        # else:
        #     reduced_loss = loss

        losses.update(to_python_float(loss), imgs.size(0))
        top1.update(to_python_float(acc1), imgs.size(0))
        top5.update(to_python_float(acc5), imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if index % 2000 == 0:
            progress.display(index)

        # torch.cuda.synchronize()
        del imgs, target, loss, output
    return losses, top1, top5


def validate(val_loader, net, criterion, batch_size=256,
             distributed=False, world_size=1):
    """Validate epoch.
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    val_loader_len = math.ceil(val_loader._size / batch_size)
    progress = ProgressMeter(
        val_loader_len, [batch_time, losses, top1, top5], prefix='Test: ')

    # Switch to evaluate mode.
    net.eval()
    with torch.no_grad():
        end = time.time()
        for index, data in enumerate(val_loader):
            imgs = data[0]["data"].cuda(non_blocking=True)
            target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
            output = net(imgs)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # if distributed:
            #     reduced_loss = reduce_tensor(loss, world_size)
            #     acc1 = reduce_tensor(acc1, world_size)
            #     acc5 = reduce_tensor(acc5, world_size)
            # else:
            #     reduced_loss = loss

            losses.update(to_python_float(loss), imgs.size(0))
            top1.update(to_python_float(acc1), imgs.size(0))
            top5.update(to_python_float(acc5), imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if index % 2000 == 0:
                progress.display(index)

            del imgs, target, loss, output

    return losses, top1, top5


# best_acc1 = 0.0
sota = {}
sota["epoch"] = -1
sota["top1"] = 0.0


@timethis
def main():
    configs = config.Config()
    configs.print_config(print)
    configs_dict = configs.get_config()
    distributed = configs_dict["distributed"]
    ngpus_per_node = torch.cuda.device_count()

    if distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, configs_dict))
    else:
        print("Distributed seeting should be set true.")
        sys.exit(-1)


def main_worker(rank_index, ngpus_per_node, configs_dict):
    exp = configs_dict["experiment_index"]
    cuda_id = configs_dict["cuda"]
    num_workers = configs_dict["num_workers"]
    seed = configs_dict["seed"]
    n_epochs = configs_dict["n_epochs"]
    log_dir = configs_dict["log_dir"]
    model_dir = configs_dict["model_dir"]
    batch_size = configs_dict["batch_size"]
    learning_rate = configs_dict["learning_rate"]
    dataset_name = configs_dict["dataset"]
    re_size = configs_dict["re_size"]
    input_size = configs_dict["input_size"]
    backbone = configs_dict["backbone"]
    eval_frequency = configs_dict["eval_frequency"]
    resume = configs_dict["resume"]
    opt = configs_dict["optimizer"]
    initialization = configs_dict["initialization"]
    weight_decay = configs_dict["weight_decay"]
    dropout = configs_dict["dropout"]
    conv_bias = configs_dict["conv_bias"]
    linear_bias = configs_dict["linear_bias"]
    freeze = configs_dict["freeze"]
    data_dir = configs_dict["data_dir"]
    # local_rank = configs_dict["local_rank"]
    world_size = configs_dict["world_size"]
    momentum = configs_dict["momentum"]
    prof = configs_dict["prof"]
    distributed = configs_dict["distributed"]
    dist_url = configs_dict["dist_url"]

    global sota

    if distributed:
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=dist_url,
                                             world_size=world_size,
                                             rank=rank_index)
        print("=> Initial at process {}".format(rank_index))
        # try:
        #     from apex.parallel import DistributedDataParallel as DDP
        # except ImportError:
        #     raise ImportError("Please install apex from \
        #                       https://www.github.com/nvidia/apex \
        #                       to run this example.")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print = init_logging(log_dir, exp).info
    init_environment(seed=seed, cuda_id=rank_index + cuda_id, _print=_print)

    # Only do in rank 0 process.
    if rank_index == 0:
        tf_log = os.path.join(log_dir, exp)
        writer = SummaryWriter(log_dir=tf_log)

    if dataset_name == "ImageNet":
        _print(">> Using dali to accelebrate.")

        batch_size_per_gpu = int(batch_size / world_size)
        train_loader = imagenet_dali.get_imagenet_iter_dali(
            mode="train", data_dir=data_dir, batch_size=batch_size_per_gpu,
            num_threads=num_workers, crop=re_size, world_size=world_size,
            local_rank=0)

        val_loader = imagenet_dali.get_imagenet_iter_dali(
            mode="val", data_dir=data_dir, batch_size=batch_size_per_gpu,
            num_threads=num_workers, crop=re_size,
            world_size=world_size, local_rank=0)

        num_classes = 1000
        input_channel = 3
        metric_priority = "top1"
    else:
        _print("Need dataset")
        sys.exit(-1)

    pretrained = False
    if initialization == "pretrained":
        pretrained = True
    _print("Initialization with {}".format(initialization))

    # _print("Using pretrained: {}".format(pretrained))
    # _print("Using dropout: {}".format(dropout))
    # _print("Using conv_bias: {}".format(conv_bias))
    # _print("Using linear_bias: {}".format(linear_bias))
    net = model.Network(backbone=backbone, num_classes=num_classes,
                        input_channel=input_channel, pretrained=pretrained,
                        dropout=dropout, conv_bias=conv_bias,
                        linear_bias=linear_bias)

    if distributed:
        # net = DDP(net, delay_allreduce=True)
        net.cuda()
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[0])

    if initialization not in ("default", "pretrained"):
        net = init_weights(net, initialization, _print)

    # if freeze:
    #     _print("Freeze model")
    #     net = freeze_model(net, _print)

    _print(">> Dataset:{} - Input size: {}".format(dataset_name, input_size))

    criterion = nn.CrossEntropyLoss()

    if opt == "SGD":
        _print("Using optimizer SGD with lr:{:.4f}".format(learning_rate))
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
    # elif optimizer == "Adam":
    #     _print("Using optimizer Adam with lr:{:.4f}".format(learning_rate))
    #     opt = torch.optim.Adam(net.parameters(), lr=learning_rate,
    #                            betas=(0.9, 0.999), eps=1e-08,
    #                            weight_decay=weight_decay, amsgrad=True)
    else:
        _print("Need optimizer")
        sys.exit(-1)

    start_epoch = 0
    if resume != "None":
        _print("Resume from model at epoch {}".format(resume))
        resume_path = os.path.join(model_dir, str(exp), str(resume))
        ckpt = torch.load(resume_path)
        net.load_state_dict(ckpt)
        start_epoch = resume + 1
    else:
        _print("Train from scrach!!")

    for epoch in range(start_epoch, n_epochs):
        losses, top1, top5 = train(train_loader, net, criterion,
                                   optimizer, epoch,
                                   batch_size=batch_size_per_gpu,
                                   prof=prof, distributed=distributed,
                                   world_size=world_size,
                                   init_lr=learning_rate)

        if rank_index == 0:
            writer.add_scalar("Lr", get_lr(optimizer), epoch)
            writer.add_scalar("Loss/train/", losses.avg, epoch)
            writer.add_scalar("Acc1/train/", top1.avg, epoch)
            writer.add_scalar("Acc5/train/", top5.avg, epoch)
            _print("Epoch:{} - train loss: {:.3f}".format(epoch, losses.avg))
            _print("Epoch:{} - train acc1: {:.3f}; acc5: {:.3f}".
                   format(epoch, top1.avg, top5.avg))

        if freeze or epoch % eval_frequency or epoch == n_epochs-1:
            losses, top1, top5 = validate(val_loader, net, criterion,
                                          batch_size=batch_size_per_gpu,
                                          distributed=distributed,
                                          world_size=world_size)

            if rank_index == 0:
                _print("Epoch:{} - Val top1: {:.4f}".format(epoch, top1.avg))
                _print("Epoch:{} - Val top5: {:.4f}".format(epoch, top5.avg))
                writer.add_scalar("Loss/val/", losses.avg, epoch)
                writer.add_scalar("Acc1/val/", top1.avg, epoch)
                writer.add_scalar("Acc5/val/", top5.avg, epoch)

            # if metric_priority == "top1":
            #     metric = top1
            #     acc = top1.avg
            # elif metric_priority == "top5":
            #     metric = top5
            #     acc = top5.avg

            # Val acc and only save in rank 0.
            if rank_index == 0 and top1.avg > sota["top1"]:
                sota["top1"] = top1.avg
                sota["epoch"] = epoch
                model_path = os.path.join(model_dir, str(exp), str(epoch))
                _print("Save model in {}".format(model_path))
                net_state_dict = net.state_dict()
                torch.save(net_state_dict, model_path)

            val_loader.reset()

        train_loader.reset()

    _print("Finish Training")
    _print("Best epoch {} with {} on Val: {:.4f}".
           format(sota["epoch"], metric_priority, sota[metric_priority]))


if __name__ == "__main__":
    main()
