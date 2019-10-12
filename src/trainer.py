# copyright 2019 jiaxin zhuang
#
#
# ?? license
# ==============================================================================
"""Trainer.

Erase image by our defined loss

"""
import sys
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from PIL import Image

from utils.function import init_logging, init_environment, recreate_image, \
        get_lr, save_image, dataname_2_save
# preprocess_image

import config
import dataset
import model
import loss

configs = config.Config()
configs_dict = configs.get_config()
exp = configs_dict["experiment_index"]
cuda_id = configs_dict["cuda"]
num_workers = configs_dict["num_workers"]
seed = configs_dict["seed"]
n_epochs = configs_dict["n_epochs"]
log_dir = configs_dict["log_dir"]
model_dir = configs_dict["model_dir"]
generated_dir = configs_dict["generated_dir"]
batch_size = configs_dict["batch_size"]
learning_rate = configs_dict["learning_rate"]
dataset_name = configs_dict["dataset"]
re_size = configs_dict["re_size"]
input_size = configs_dict["input_size"]
backbone = configs_dict["backbone"]
eval_frequency = configs_dict["eval_frequency"]
resume = configs_dict["resume"]
optimizer = configs_dict["optimizer"]
selected_layer = configs_dict["selected_layer"]
selected_filter = configs_dict["selected_filter"]
alpha = configs_dict["alpha"]
weight_decay = configs_dict["weight_decay"]
class_index = configs_dict["class_index"]
num_class = configs_dict["num_class"]
mode = configs_dict["mode"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

init_environment(seed=seed, cuda_id=cuda_id)
_print = init_logging(log_dir, exp).info
configs.print_config(_print)
tf_log = os.path.join(log_dir, exp)
writer = SummaryWriter(log_dir=tf_log)
generated_dir = os.path.join(generated_dir, exp)

_print("Save generated on {}".format(generated_dir))
_print("Using device {}".format(device))
_print("Alpha is {}".format(alpha))

if dataset_name == "mnist":
    mean, std = (0.1307,), (0.3081,)
    reverse_mean = (-0.1307,)
    reverse_std = (1/0.3081,)
    # train_transform = transforms.Compose([
    #     transforms.Resize((re_size, re_size), interpolation=Image.BILINEAR),
    #     transforms.ToTensor(),
    #     transforms.Normalize((mean, ) , (std, ))
    # ])
    # val_transform = transforms.Compose([
    #     transforms.Resize((re_size, re_size), interpolation=Image.BILINEAR),
    #     transforms.ToTensor(),
    #     transforms.Normalize((mean, ), (std, ))
    # ])
    train_transform = None
    val_transform = None
    trainset = dataset.MNIST(root="./data/", is_train=True,
                             transform=train_transform)
    valset = dataset.MNIST(root="./data/", is_train=False,
                           transform=val_transform)
elif dataset_name == "CUB":
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    train_transform = transforms.Compose([
        transforms.Resize((re_size, re_size), interpolation=Image.BILINEAR),
        # transforms.RandomCrop(input_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((re_size, re_size), interpolation=Image.BILINEAR),
        # transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    trainset = dataset.CUB(root="./data/", is_train=True,
                           transform=train_transform)
    valset = dataset.CUB(root="./data/", is_train=False,
                         transform=test_transform)
    num_classes = 200
    input_channel = 3
else:
    _print("Need dataset")
    sys.exit(-1)

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,\
#                                           shuffle=False,
#                                           num_workers=num_workers)
# valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, \
#                                         shuffle=False,
#                                         num_workers=num_workers)

_print(">> Dataset:{} - Input size: {}".format(dataset_name, input_size))


trainset.set_data(class_index, num_class)
images = []
labels = []
imgs_path = []
for img, label, img_path in trainset:
    images.append(img.unsqueeze(0))
    labels.append(label)
    imgs_path.append(img_path)

original_images = torch.cat(images, dim=0).to(device)
processed_images = original_images.clone().detach().requires_grad_(True)
processed_images = processed_images.to(device)


# processed_image = preprocess_image(images, mean=mean, std=std,
#                                    resize_im=False,
#                                    resize=re_size, device=device)
# original_image = processed_image.clone().detach()

net = model.Network(backbone=backbone, num_classes=num_classes)
net.to(device)

_print("Loss using mode: {}".format(mode))
criterion = loss.FileterLoss(net, selected_layer, selected_filter, mode)

# Define optimizer for the image
# Earlier layers need higher learning rates to visualize whereas layer layers
# need less
scheduler = None
if optimizer == "SGD":
    _print("Using optimizer SGD with lr:{:.4f}".format(learning_rate))
    opt = torch.optim.SGD([processed_images], lr=learning_rate, momentum=0.9,
                          weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.1, patience=1000, verbose=True,
                threshold=1e-4)
elif optimizer == "Adam":
    _print("Using optimizer Adam with lr:{:.4f}".format(learning_rate))
    opt = torch.optim.Adam([processed_images], lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=weight_decay, amsgrad=False)
else:
    _print("Optimizer not available")
    sys.exit(-1)

start_epoch = 0
if resume:
    resume_exp = resume.split("-")[0]
    resume_epoch = resume.split("-")[1]
    _print("Resume from model from exp: {} at epoch {}".
           format(resume_exp, resume_epoch))
    resume_path = os.path.join(model_dir, str(resume_exp), str(resume_epoch))
    ckpt = torch.load(resume_path)
    net.load_state_dict(ckpt)
else:
    _print("Train from scrach!!")


desc = "Exp-{}-Train".format(exp)
sota = {"epoch": -1, "acc": -1.0}

original_images = original_images.to(device)

losses = []
for epoch in range(n_epochs):
    opt.zero_grad()
    selected_filter_loss, rest_fileter_loss = criterion(processed_images,
                                                        original_images)
    loss = selected_filter_loss + alpha * rest_fileter_loss
    loss.backward()
    opt.step()
    losses.append(loss.item())
    writer.add_scalar("Loss/selected_filter_loss", selected_filter_loss.item(),
                      epoch)
    writer.add_scalar("Loss/rest_fileter_loss", rest_fileter_loss.item(),
                      epoch)
    train_loss = loss.item()
    if scheduler is not None:
        scheduler.step(train_loss)

    writer.add_scalar("Lr", get_lr(opt), epoch)
    writer.add_scalar("Loss/total/", train_loss, epoch)
    _print("selected_fileter_loss: {:.4f}".format(selected_filter_loss.item()))
    _print("rest_fileter_loss: {:.4f}".format(rest_fileter_loss.item()))
    _print("Epoch:{} - train loss: {:.4f}".format(epoch, train_loss))

    if epoch % eval_frequency == 0:
        saved_dir = os.path.join(generated_dir, str(epoch))
        os.makedirs(saved_dir, exist_ok=True)
        saved_paths = dataname_2_save(imgs_path, saved_dir)
        processed_images_cpu = processed_images.detach().cpu().numpy()
        for img, save_path in zip(processed_images_cpu, saved_paths):
            recreate_im = recreate_image(img,
                                         reverse_mean=reverse_mean,
                                         reverse_std=reverse_std)
            save_image(recreate_im, save_path)
            _print("save generated image in {}".format(save_path))
            # writer.add_image("recreate_image", recreate_im, epoch,
            #                  dataformats='HWC')

_print("Finish Training")
