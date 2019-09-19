import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

import sklearn
from sklearn.metrics import accuracy_score
from PIL import Image

from utils.function import init_logging, init_environment
import config
import dataset
import model

configs = config.Config()
configs_dict = configs.get_config()
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
optimizer = configs_dict["optimizer"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

init_environment(seed=seed, cuda_id=cuda_id)
_print = init_logging(log_dir, exp).info
configs.print_config(_print)
tf_log = os.path.join(log_dir, exp)
writer = SummaryWriter(log_dir=tf_log)

_print("Using device {}".format(device))

if dataset_name == "mnist":
    mean, std = 0.1307, 0.3081
    train_transform = transforms.Compose([
        transforms.Resize((re_size, re_size), interpolation=Image.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((mean, ) , (std, ))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((re_size, re_size), interpolation=Image.BILINEAR),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((mean, ), (std, ))
    ])
    trainset = dataset.MNIST(root="./data/", is_train=True, transform=train_transform)
    valset = dataset.MNIST(root="./data/", is_train=False, transform=val_transform)


else:
    _print("Need dataset")
    sys.exit(-1)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,\
                                          shuffle=True, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, \
                                        shuffle=False, num_workers=num_workers)


net = model.Network(backbone=backbone)
net.to(device)


_print(">> Dataset:{} - Input size: {}".format(dataset_name, input_size))

criterion = nn.CrossEntropyLoss()


if optimizer == "SGD":
    _print("Using optimizer SGD with lr:{:.4f}".format(learning_rate))
    opt = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
elif optimizer == "Adam":
    _print("Using optimizer Adam with lr:{:.4f}".format(learning_rate))
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=0, amsgrad=False)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.1, patience=10, verbose=True,
            threshold=1e-4)

start_epoch = 0
if resume:
    _print("Resume from model at epoch {}".format(resume))
    resume_path = os.path.join(model_dir, str(exp), str(resume))
    ckpt = torch.load(resume_path)
    net.load_state_dict(ckpt)
    start_epoch = resume + 1
else:
    _print("Train from scrach!!")


desc = "Exp-{}-Train".format(exp)
sota = dict()
sota["epoch"] = start_epoch
sota["acc"] = -1.0

for epoch in range(start_epoch, n_epochs):
    net.train()
    losses = []
    for _, (data, target) in enumerate(tqdm(trainloader, ncols=70, desc=desc)):
        data, target = data.to(device), target.to(device)
        predict = net(data)
        opt.zero_grad()
        loss = criterion(predict, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    train_avg_loss = np.mean(losses)

    if scheduler is not None:
        scheduler.step(train_avg_loss)

    writer.add_scalar("Loss/train/", train_avg_loss, epoch)
    _print("Epoch:{} - train loss: {:.3f}".format(epoch, train_avg_loss))

    if epoch % eval_frequency:
        y_true = []
        y_pred = []
        for _, (data, target) in enumerate(tqdm(trainloader, ncols=70, desc="train")):
            data = data.to(device)
            predict = torch.argmax(net(data), dim=1).cpu().data.numpy()
            y_pred.extend(predict)
            target = target.cpu().data.numpy()
            y_true.extend(target)

        acc = accuracy_score(y_true, y_pred)
        _print("Epoch:{} - train acc: {:.4f}".format(epoch, acc))
        writer.add_scalar("Acc/train/", train_avg_loss, epoch)

        y_true = []
        y_pred = []
        for _, (data, target) in enumerate(tqdm(valloader, ncols=70, desc="val")):
            data = data.to(device)
            predict = torch.argmax(net(data), dim=1).cpu().data.numpy()
            y_pred.extend(predict)
            target = target.cpu().data.numpy()
            y_true.extend(target)

        acc = accuracy_score(y_true, y_pred)
        _print("Epoch:{} - val acc: {:.4f}".format(epoch, acc))
        writer.add_scalar("Acc/train/", train_avg_loss, epoch)

        # Val acc
        if acc > sota["acc"]:
            sota["acc"] = acc
            sota["epoch"] = epoch
            model_path = os.path.join(model_dir, str(exp), str(epoch))
            _print("Save model in {}".format(model_path))
            net_state_dict = net.state_dict()
            torch.save(net_state_dict, model_path)

_print("Finish Training")
_print("Best epoch {} with Val: {:.4f}".format(sota["epoch"], sota["acc"]))
