# copyright 2019 jiaxin zhuang
#
# # ?? license
# ==============================================================================
"""Trainer.

Erase image by our defined loss

"""
import sys
import os
import torch
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from collections import defaultdict
import pickle
from PIL import Image
import shutil
import numpy as np

from utils.function import init_logging, init_environment, recreate_image, \
        get_lr, save_image, dataname_2_save, timethis
from utils.early_stopping import EarlyStopping
from utils.visualizations.visualize import visualize
from utils.function import zscore
from myMetric import mMetric_v3
from model import replace_layer
from loss import FilterLoss
from datasets import imagenet

import config
import dataset
import model


@timethis
def main():
    configs = config.Config()
    configs_dict = configs.get_config()
    exp = configs_dict["experiment_index"]
    cuda_id = configs_dict["cuda"]
    seed = configs_dict["seed"]
    n_epochs = configs_dict["n_epochs"]
    log_dir = configs_dict["log_dir"]
    model_dir = configs_dict["model_dir"]
    generated_dir = configs_dict["generated_dir"]
    batch_size = configs_dict["batch_size"]
    learning_rate = configs_dict["learning_rate"]
    dataset_name = configs_dict["dataset"]
    data_dir = configs_dict["data_dir"]
    re_size = configs_dict["re_size"]
    input_size = configs_dict["input_size"]
    backbone = configs_dict["backbone"]
    eval_frequency = configs_dict["eval_frequency"]
    resume = configs_dict["resume"]
    optimizer = configs_dict["optimizer"]
    selected_layer = configs_dict["selected_layer"]
    selected_filter = configs_dict["selected_filter"]
    alpha = configs_dict["alpha"]
    beta = configs_dict["beta"]
    weight_decay = configs_dict["weight_decay"]
    class_index = configs_dict["class_index"]
    num_class = configs_dict["num_class"]
    mode = configs_dict["mode"]
    inter = configs_dict["inter"]
    rho = configs_dict["rho"]
    regularization = configs_dict["regularization"]
    regular_ex = configs_dict["regular_ex"]
    gamma = configs_dict["gamma"]
    smoothing = configs_dict["smoothing"]
    delta = configs_dict["delta"]
    img_index = configs_dict["img_index"]
    rescale = configs_dict["rescale"]
    guidedReLU = configs_dict["guidedReLU"]
    defensed = configs_dict["defensed"]
    avg = configs_dict["avg"]
    initialization = configs_dict["initialization"]
    use_scheduler = configs_dict["scheduler"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _print = init_logging(log_dir, exp).info
    configs.print_config(_print)
    init_environment(seed=seed, cuda_id=cuda_id, _print=_print)
    # tf_log = os.path.join(log_dir, exp)
    # writer = SummaryWriter(log_dir=tf_log)
    generated_dir = os.path.join(generated_dir, exp)
    # if os.path.exists(generated_dir):
    #     shutil.rmtree(generated_dir)
    # os.makedirs(generated_dir)

    _print("Save generated on {}".format(generated_dir))
    _print("Using device {}".format(device))
    _print("Alpha is {}, Beta is {}, Gamma is {}".format(alpha, beta, gamma))
    _print("Whether to use inter: {} with rho: {}".format(inter, rho))
    _print("Whether to use regularization: {}".format(regularization))
    _print("Whether to use Smoothing: {} with delta: {}".format(smoothing,
                                                                delta))

    if dataset_name == "mnist":
        mean, std = (0.1307,), (0.3081,)
        # mean, std = 0.1307, 0.3081
        reverse_mean = (-0.1307,)
        reverse_std = (1/0.3081,)
        train_transform = transforms.Compose([
            transforms.Resize((re_size, re_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        # val_transform = transforms.Compose([
        #     transforms.Resize((re_size, re_size),
        #     interpolation=Image.BILINEAR),
        #     transforms.ToTensor(),
        #     transforms.Normalize((mean, ), (std, ))
        # ])
        # val_transform = None
        trainset = dataset.MNIST(root="./data/", is_train=True,
                                 transform=train_transform)
        # valset = dataset.MNIST(root="./data/", is_train=False,
        #                        transform=val_transform)
        num_classes = 10
        input_channel = 1
    elif dataset_name == "CUB":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        reverse_mean = [-0.485, -0.456, -0.406]
        reverse_std = [1/0.229, 1/0.224, 1/0.225]
        train_transform = transforms.Compose([
            transforms.Resize((re_size, re_size),
                              interpolation=Image.BILINEAR),
            # transforms.RandomCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        # test_transform = transforms.Compose([
        #     transforms.Resize((re_size, re_size),
        #                       interpolation=Image.BILINEAR),
        #     # transforms.CenterCrop(input_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)
        # ])
        trainset = dataset.CUB(root="./data/", is_train=True,
                               transform=train_transform)
        # valset = dataset.CUB(root="./data/", is_train=False,
        #                      transform=test_transform)
        num_classes = 200
        input_channel = 3
    elif dataset_name == "Caltech101":
        mean = [0.5495916, 0.52337694, 0.49149787]
        std = [0.3202951, 0.31704363, 0.32729807]
        reverse_mean = [-0.5495916, -0.52337694, -0.49149787]
        reverse_std = [1/0.3202951, 1/0.31704363, 1/0.32729807]
        train_transform = transforms.Compose([
            transforms.Resize((re_size, re_size),
                              interpolation=Image.BILINEAR),
            # transforms.RandomCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        # test_transform = transforms.Compose([
        #     transforms.Resize((re_size, re_size),
        #                       interpolation=Image.BILINEAR),
        #     # transforms.CenterCrop(input_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)
        # ])
        trainset = dataset.Caltech101(root="./data/", is_train=True,
                                      transform=train_transform)
        # valset = dataset.Caltech101(root="./data/", is_train=False,
        #                             transform=test_transform)
        num_classes = 101
        input_channel = 3
    elif dataset_name == "ImageNet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        reverse_mean = [-0.485, -0.456, -0.406]
        reverse_std = [1/0.229, 1/0.224, 1/0.225]
        train_transform = transforms.Compose([
            transforms.Resize((re_size, re_size),
                              interpolation=Image.BILINEAR),
            # transforms.RandomCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        # test_transform = transforms.Compose([
        #     transforms.Resize((re_size, re_size),
        #                       interpolation=Image.BILINEAR),
        #     transforms.CenterCrop(input_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)
        # ])
        trainset = imagenet.ImageNet(root=data_dir, is_train=True,
                                     transform=train_transform)
        # valset = imagenet.ImageNet(root="./data/", is_train=False,
        #                            transform=test_transform)
        num_classes = 1000
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
# When single images used, batch_size has to be one
    if batch_size == 1:
        _print("Used single image mode.")
        images = [images[img_index]]
        labels = [labels[img_index]]
        imgs_path = [imgs_path[img_index]]

    original_images = torch.cat(images, dim=0).to(device)
    processed_images = original_images.clone().detach().requires_grad_(True)
    processed_images = processed_images.to(device)


# processed_image = preprocess_image(images, mean=mean, std=std,
#                                    resize_im=False,
#                                    resize=re_size, device=device)
# original_image = processed_image.clone().detach()

    if initialization == "pretrained":
        pretrained = True
    else:
        pretrained = False
    net = model.Network(backbone=backbone, num_classes=num_classes,
                        selected_layer=selected_layer, guidedReLU=guidedReLU,
                        pretrained=pretrained, input_channel=input_channel)

    if avg:
        _print("Replace maxpool with avgpool.")
        net = replace_layer(net, keys=["AVG"])

    net.to(device)
    net.eval()

    _print("Loss using mode: {}".format(mode))
    if backbone == "vgg16":
        selected_layer = int(selected_layer)
    criterion = FilterLoss(net, selected_layer, selected_filter, mode,
                           inter=inter, rho=rho,
                           regularization=regularization, defensed=defensed,
                           smoothing=smoothing, p=regular_ex, _print=_print)

# Define optimizer for the image
# Earlier layers need higher learning rates to visualize whereas layer layers
# need less
    scheduler = None
    if optimizer == "SGD":
        _print("Using optimizer SGD with lr:{:.4f}".format(learning_rate))
        opt = torch.optim.SGD([processed_images], lr=learning_rate,
                              momentum=0.9,
                              weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode='min', factor=0.5, patience=200, verbose=True,
                    threshold=1e-4)
    elif optimizer == "Adam":
        _print("Using optimizer Adam with lr:{:.4f}".format(learning_rate))
        opt = torch.optim.Adam([processed_images], lr=learning_rate,
                               betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=weight_decay, amsgrad=False)
    else:
        _print("Optimizer not available")
        sys.exit(-1)

    if not use_scheduler:
        scheduler = None

    start_epoch = 0
    if resume and backbone in ["vgg11", "vgg16"]:
        resume_exp = resume.split("-")[0]
        resume_epoch = resume.split("-")[1]
        _print("Resume from model from exp: {} at epoch {}".
               format(resume_exp, resume_epoch))
        resume_path = os.path.join(model_dir, str(resume_exp),
                                   str(resume_epoch))
        ckpt = torch.load(resume_path)
        net.load_state_dict(ckpt, strict=False)
    else:
        _print("Train from scrach!!")

    earlystopping = EarlyStopping(mode="min", min_delta=1e-5, patience=100)

    original_images = original_images.to(device)

    # GuildeReLU
    net.set_guildedReLU(guidedReLU)
    _print(net)

    # original activation maps.
    criterion.load_ori_activation(original_images)

    statistics = defaultdict(list)
    best = {"loss": float('inf')}
    for epoch in range(start_epoch, n_epochs):
        opt.zero_grad()

        # Need for defensed.
        if len(criterion.original_activation_maps) == 0:
            criterion.load_ori_activation(original_images)

        selected_filter_loss, rest_fileter_loss, regularization_loss,  \
            smoothing_loss, rest_filter_loss_interact = \
            criterion(processed_images)
        # if alpha != 0 and (1-alpha) != 0:
        # use beat to omit gradient from rest_filter_loss
        if criterion.inter:
            loss = alpha * selected_filter_loss / criterion.ori_sel_l2 + \
                beta * rest_filter_loss_interact / criterion.ori_rest_l1 + \
                gamma * regularization_loss + delta * smoothing_loss
        else:
            loss = alpha * selected_filter_loss / criterion.ori_sel_l2 + \
                beta * rest_fileter_loss / criterion.ori_rest_l1 + \
                gamma * regularization_loss + delta * smoothing_loss
        loss.backward()

        # Clip gradient using maximum value
        # if clip_grad:
        # .nn.utils.clip_grad_norm_(net.parameters(), max_norm=1,
        #                                    norm_type=torch._six.inf)

        # writer.add_histogram("Processed_images", processed_images.clone().
        #                      cpu().data.numpy(), epoch)
        # writer.add_histogram("Processed_images_grad", processed_images.grad.
        #                      clone().cpu().data.numpy(), epoch)
        opt.step()
        train_loss = loss.item()
        if scheduler is not None:
            scheduler.step(train_loss)

        # early stopping
        is_break = False
        if get_lr(opt) < 1:
            if earlystopping.step(torch.tensor(train_loss)):
                is_break = True

        # find best
        if train_loss < best["loss"]:
            ori_activation_maps = net.get_activation_maps(original_images,
                                                          selected_layer)[0]
            opt_activation_maps = net.get_activation_maps(processed_images,
                                                          selected_layer)[0]
            metric_statistic = mMetric_v3(ori_activation_maps,
                                          opt_activation_maps, selected_filter)
            best["epoch"] = epoch
            best["loss"] = train_loss
            best["processed_images"] = processed_images.detach().clone()
            best["epoch"] = epoch
            best["selected_filter_loss"] = selected_filter_loss.item()
            best["rest_fileter_loss"] = rest_fileter_loss.item()
            best["regularization_loss"] = regularization_loss.item()
            best["rest_filter_loss_interact"] = \
                rest_filter_loss_interact.item()
            best["metric_statistic"] = metric_statistic

        if epoch % eval_frequency == 0:
            ori_activation_maps = net.get_activation_maps(original_images,
                                                          selected_layer)[0]
            opt_activation_maps = net.get_activation_maps(processed_images,
                                                          selected_layer)[0]
            metric_statistic = mMetric_v3(ori_activation_maps,
                                          opt_activation_maps,
                                          selected_filter, _print=_print)
            statistics["epoch"].append(epoch)
            statistics["selected_filter_loss"].append(
                selected_filter_loss.item())
            statistics["rest_fileter_loss"].append(
                rest_fileter_loss.item())
            statistics["regularization_loss"].append(
                regularization_loss.item())
            statistics["rest_filter_loss_interact"].append(
                rest_filter_loss_interact.item())
            statistics["losses"].append(train_loss)
            statistics["metric_statistic"].append(metric_statistic)
            statistics["lr"].append(get_lr(opt))

            saved_paths = dataname_2_save(imgs_path, generated_dir, epoch)
            processed_images_cpu = processed_images.detach().cpu().numpy()
            for img, save_path in zip(processed_images_cpu, saved_paths):
                recreate_im = recreate_image(img,
                                             reverse_mean=reverse_mean,
                                             reverse_std=reverse_std,
                                             rescale=rescale)
                save_image(recreate_im, save_path)

        # In order to save last epoch
        if epoch+1 == n_epochs:
            # statistics
            for img_path in imgs_path:
                statistics["best"] = [best]
                name = os.path.splitext(img_path)[0].split("/")[-1]
                statistics_path = os.path.join(generated_dir,
                                               name + "_statistics.pl")
                with open(statistics_path, "wb") as handle:
                    pickle.dump(statistics, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

            # [batch_size, channel, height, width]
            epoch = best["epoch"]
            processed_images_cpu = best["processed_images"]

            saved_paths = dataname_2_save(imgs_path, generated_dir,
                                          str(epoch)+"_best")
            processed_images_cpu = processed_images_cpu.cpu().numpy()

            for img, save_path in zip(processed_images_cpu, saved_paths):
                recreate_im = recreate_image(img,
                                             reverse_mean=reverse_mean,
                                             reverse_std=reverse_std,
                                             rescale=rescale)
                save_image(recreate_im, save_path)

            print(save_path)
            opt_image = Image.open(save_path)
            if opt_image.mode == 'L':
                opt_image = np.expand_dims(opt_image, axis=2)
            opt_image = np.expand_dims(opt_image, axis=0).astype(np.float32)
            opt_image = opt_image / 255.0
            saved_opt_img = zscore(opt_image, mean, std)
            saved_opt_img = torch.from_numpy(saved_opt_img).to(device)

            ori_activation_maps = net.get_activation_maps(original_images,
                                                          selected_layer)
            opt_activation_maps = net.get_activation_maps(saved_opt_img,
                                                          selected_layer)

            metric_statistic = mMetric_v3(ori_activation_maps[0],
                                          opt_activation_maps[0],
                                          selected_filter, _print=_print)
            print(metric_statistic)
            visualize(ori_activation_maps, opt_activation_maps,
                      img_index=img_index, layer_name=selected_layer,
                      backbone=backbone, num_class=num_class,
                      exp=exp, imgs_path=imgs_path)

            _print(">>> EarlyStopping at epoch: {} <<<".format(epoch))
            break

    _print("Finish Training")


if __name__ == "__main__":
    main()
