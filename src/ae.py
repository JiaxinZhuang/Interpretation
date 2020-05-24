"""Autoencoder.
"""

import sys
import os
import torch
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from datasets import DTD, skin7
from models.Autoencoder import Autoencoder
import config
from model import init_weights
from utils.function import init_logging, init_environment, get_lr, timethis,\
    to_python_float
# ProgressMeter
from utils.metric import AverageMeter, accuracy


def train(loader, net, cls_criterion, rec_criterion, optimizer,
          batch_size, print_freq, epoch, device, alpha, beta):
    losses = AverageMeter('losses', ':.4e')
    cls_losses = AverageMeter('cls_losses', ':.4e')
    rec_losses = AverageMeter('rec_losses', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    batch_time = AverageMeter('Time', ':6.3f')
    date_time = AverageMeter('Date', ':6.3f')

    net.train()
    end = time.time()
    for index, (imgs, target) in enumerate(loader):
        # measure imgs loading time
        date_time.update(time.time() - end)

        imgs = imgs.to(device)
        target = target.to(device)
        output, preds = net(imgs)
        classification_loss = cls_criterion(preds, target)
        reconstruction_loss = rec_criterion(output, imgs)
        loss = alpha * classification_loss + beta * reconstruction_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1 = accuracy(preds, target, topk=(1,))

        losses.update(to_python_float(loss), imgs.size(0))
        cls_losses.update(to_python_float(classification_loss),
                          imgs.size(0))
        rec_losses.update(to_python_float(reconstruction_loss),
                          imgs.size(0))
        top1.update(to_python_float(acc1), imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        del imgs, target, preds, output

        # if index % print_freq == 0:
        #     progress.display(index)

    return losses, cls_losses, rec_losses, top1


def validate(loader, net, cls_criterion, rec_criterion, batch_size,
             print_freq, epoch, device, alpha, beta):
    """Validate.
    """
    losses = AverageMeter('losses', ':.4e')
    cls_losses = AverageMeter('cls_losses', ':.4e')
    rec_losses = AverageMeter('rec_losses', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    batch_time = AverageMeter('Time', ':6.3f')
    date_time = AverageMeter('Date', ':6.3f')

    # val_loader_len = math.ceil(len(loader) / batch_size)
    # progress = ProgressMeter(
    #     val_loader_len, [batch_time, losses, cls_losses, rec_losses,
    #                      top1], prefix='Test: ')

    with torch.no_grad():
        net.eval()
        end = time.time()
        for index, (imgs, target) in enumerate(loader):
            # measure imgs loading time
            date_time.update(time.time() - end)

            imgs = imgs.to(device)
            target = target.to(device)
            output, preds = net(imgs)
            classification_loss = cls_criterion(preds, target)
            reconstruction_loss = rec_criterion(output, imgs)
            loss = alpha * classification_loss + beta * reconstruction_loss

            acc1 = accuracy(preds, target, topk=(1,))
            top1.update(to_python_float(acc1), imgs.size(0))
            losses.update(to_python_float(loss), imgs.size(0))
            cls_losses.update(to_python_float(classification_loss),
                              imgs.size(0))
            rec_losses.update(to_python_float(reconstruction_loss),
                              imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            del imgs, target, preds, output

            # if index % print_freq == 0:
            #     progress.display(index)
    return losses, cls_losses, rec_losses, top1


@timethis
def main():
    configs = config.Config()
    configs_dict = configs.get_config()
    exp = configs_dict["experiment_index"]
    cuda_id = configs_dict["cuda"]
    num_workers = configs_dict["num_workers"]
    seed = configs_dict["seed"]
    n_epochs = configs_dict["n_epochs"]
    data_dir = configs_dict["data_dir"]
    log_dir = configs_dict["log_dir"]
    model_dir = configs_dict["model_dir"]
    batch_size = configs_dict["batch_size"]
    learning_rate = configs_dict["learning_rate"]
    dataset_name = configs_dict["dataset"]
    # input_size = configs_dict["input_size"]
    input_size = configs_dict["input_size"]
    eval_frequency = configs_dict["eval_frequency"]
    # resume = configs_dict["resume"]
    opt = configs_dict["optimizer"]
    print_freq = configs_dict["print_freq"]
    initialization = configs_dict["initialization"]
    alpha = configs_dict["alpha"]
    beta = configs_dict["beta"]
    embedding_size = configs_dict["embedding_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _print = init_logging(log_dir, exp).info
    configs.print_config(_print)
    init_environment(seed=seed, cuda_id=cuda_id, _print=_print)
    tf_log = os.path.join(log_dir, exp)
    writer = SummaryWriter(log_dir=tf_log)

    if dataset_name == "DTD":
        mean = [0.482, 0.482, 0.482]
        std = [0.241, 0.241, 0.241]
        train_transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = DTD.DTD(root=data_dir, is_train=True,
                           transform=train_transform)
        num_classes = 47
        # input_channel = 3
    elif dataset_name == "Skin7":
        mean = [0.7626, 0.5453, 0.5714]
        std = [0.1404, 0.1519, 0.1685]
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        val_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        trainset = skin7.Skin7(root=data_dir, train=True,
                               transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers)
        valset = skin7.Skin7(root=data_dir, train=False,
                             transform=val_transform)
        valloader = torch.utils.data.DataLoader(valset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers)
        num_classes = 7
    else:
        _print("NO dataset.")
        sys.exit(-1)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=True)

    net = Autoencoder(embedding_size=embedding_size, num_classes=num_classes)
    _print("Initialization with {}".format(initialization))
    if initialization != "pretrained":
        net = init_weights(net, initialization, _print)
    net.to(device)

    _print(">> Dataset:{} - Input size: {}".format(dataset_name, 224))

    scheduler = None
    if opt == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                     weight_decay=1e-5)
    elif opt == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                    momentum=0.9, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.1, patience=10,
                    verbose=True, threshold=1e-4)
    else:
        _print("No optimzier.")
        sys.exit(-1)

    cls_criterion = nn.CrossEntropyLoss()
    rec_criterion = nn.MSELoss()

    sota = {}
    sota["epoch"] = -1
    # sota["top1"] = -1.0
    sota["rec_loss"] = 99999999
    for epoch in range(n_epochs):
        losses, cls_losses, rec_losses, top1 = \
            train(loader=trainloader, net=net, cls_criterion=cls_criterion,
                  rec_criterion=rec_criterion, optimizer=optimizer,
                  batch_size=batch_size, print_freq=print_freq, epoch=epoch,
                  device=device, alpha=alpha, beta=beta)
        scheduler.step(losses.avg)
        writer.add_scalar("Lr", get_lr(optimizer), epoch)
        writer.add_scalar("Loss/train/", losses.avg, epoch)
        writer.add_scalar("cls_Loss/train/", cls_losses.avg, epoch)
        writer.add_scalar("res_Loss/train/", rec_losses.avg, epoch)
        writer.add_scalar("Acc1/train/", top1.avg, epoch)
        _print("Epoch: {} - train loss: {:.3f} - cls_losses: {:.3f} -\
               rec_losses: {:.3f}".format(epoch, losses.avg, cls_losses.avg,
                                          rec_losses.avg))

        if epoch % eval_frequency == 0:
            losses, cls_losses, rec_losses, top1 = \
                validate(loader=valloader, net=net,
                         cls_criterion=cls_criterion,
                         rec_criterion=rec_criterion,
                         batch_size=batch_size, print_freq=print_freq,
                         epoch=epoch, device=device, alpha=alpha, beta=beta)
            writer.add_scalar("Loss/val/", losses.avg, epoch)
            writer.add_scalar("cls_Loss/val/", cls_losses.avg, epoch)
            writer.add_scalar("res_Loss/val/", rec_losses.avg, epoch)
            writer.add_scalar("Acc1/val/", top1.avg, epoch)
            _print("Epoch: {} - val loss: {:.3f} - cls_losses: {:.3f} -\
                    rec_losses: {:.3f}".format(epoch, losses.avg,
                                               cls_losses.avg, rec_losses.avg))

        if rec_losses.avg < sota["rec_loss"]:
            # sota["top1"] = top1.avg
            sota["rec_loss"] = rec_losses.avg
            sota["epoch"] = epoch
            model_path = os.path.join(model_dir, str(exp), str(epoch))
            _print("Save model in {}".format(model_path))
            net_state_dict = net.state_dict()
            torch.save(net_state_dict, model_path)

    _print("Finish Training")


if __name__ == "__main__":
    main()
