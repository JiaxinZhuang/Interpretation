import torch
from torchvision import transforms
import os
from PIL import Image
import sklearn
from sklearn.neighbors import KNeighborsClassifier

import sys
import argparse
sys.path.append("../src/")
from datasets.skin7 import Skin7
from utils.function import get_embeddings
import utils.metric as metric
from models.Autoencoder import Autoencoder

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='kNN')
parser.add_argument("--data", default="Skin7", type=str, choices=["Skin7"])
parser.add_argument("--exp", default="051952", type=str)
parser.add_argument("--epoch", default="7", type=str)
parser.add_argument("--embedding_size", default=1000, type=int)
args = parser.parse_args()

save_model = "/home/lincolnzjx/Desktop/Interpretation/saved/models"
path = os.path.join(save_model, args.exp, args.epoch)
embedding_size = args.embedding_size
num_classes = 7
net = Autoencoder(embedding_size=embedding_size, num_classes=num_classes,
                  freeze_encoder=True)
ckpt = torch.load(path)
net.load_state_dict(ckpt)
net.cuda()
net.eval()

dataset_name = args.data
num_workers = 4
batch_size = 90

if dataset_name == "Skin7":
    mean = [0.7626, 0.5453, 0.5714]
    std = [0.1404, 0.1519, 0.1685]
    re_size = 224
    train_transform = transforms.Compose([
        transforms.Resize((re_size, re_size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    val_transform = transforms.Compose([
        transforms.Resize((re_size, re_size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    data_dir = "/media/lincolnzjx/HardDisk/Datasets"
    trainset = Skin7(root=data_dir, train=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)
    valset = Skin7(root=data_dir, train=False, transform=val_transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
else:
    print("dataset name need")
    sys.exit(-1)


train_embeddings_cpu, train_targets_cpu = get_embeddings(trainloader, net)
val_embeddings_cpu, val_targets_cpu = get_embeddings(valloader, net)
train_embeddings_cpu_np = train_embeddings_cpu.view(
    train_embeddings_cpu.size(0), -1).cpu().data.numpy()
train_targets_cpu_np = train_targets_cpu.view(-1).cpu().data.numpy()


for index in [5, 1, 9]:
    neigh = KNeighborsClassifier(n_neighbors=index, n_jobs=8)
    neigh.fit(train_embeddings_cpu_np, train_targets_cpu_np)

    val_embeddings_cpu_np = val_embeddings_cpu.view(
        val_embeddings_cpu.size(0), -1).cpu().data.numpy()
    val_targets_cpu_np = val_targets_cpu.view(-1).cpu().data.numpy()

    predict = neigh.predict(val_embeddings_cpu_np)

    acc = sklearn.metrics.accuracy_score(val_targets_cpu_np, predict)
    mca = sklearn.metrics.balanced_accuracy_score(val_targets_cpu_np, predict)
    mcp = metric.average_precision(val_targets_cpu_np, predict)
    recall = sklearn.metrics.recall_score(val_targets_cpu_np, predict,
                                          average=None)
    # mcr = mean_few_recall(trainset, recall, dataset_name=dataset_name)
    print("K@", index, " - ", acc, mca, mcp)
    # print("recall for every class", mcr)
