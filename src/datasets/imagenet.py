"""ImageNet datasets.
"""


import os
import torch
import torchvision
import numpy as np
from torchvision import transforms
from torchvision.datasets.folder import (default_loader)
import seaborn as sns

ARCHIVE_META = {
    'train': ('ILSVRC2012_img_train.tar', '1d675b47d978889d74fa0da5fadfb00e'),
    'val': ('ILSVRC2012_img_val.tar', '29b22e2961454d5413ddabcf34fc5622'),
    'devkit': ('ILSVRC2012_devkit_t12.tar.gz',
               'fa75699e90414af021442c21a62c3abf')
}

META_FILE = "meta.bin"


class ImageNet(torchvision.datasets.ImageFolder):
    """ImageNet-2012 classification datasets.

    Args:

    """
    def __init__(self, root="../data", is_train=True, transform=None,
                 _print=None):
        self.root = os.path.join(root, "ilsvrc2012")
        self.is_train = is_train
        wnid_to_classes = load_meta_file(os.path.join(self.root, META_FILE))[0]
        super(ImageNet, self).__init__(self.split_folder)

        self.transform = transform
        self._print = _print

        self.imgs_path, self.targets = self.read_path()
        # TODO:
        # 1. load meta from meta.bin
        # 2. ...
        # Inherit from ImageFolder
        self.wnids = self.classes
        self.wnids_to_idx = self.class_to_idx
        # class name
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

        self.targets_imgs_dict = {}
        self.uni_targets = np.unique(self.targets)
        targets_np = np.array(self.targets)
        for target in self.uni_targets:
            indexes = np.nonzero(target == targets_np)[0]
            self.targets_imgs_dict.update({target: indexes})

        self.data_sample = []
        self.labels_sample = []

    def __getitem__(self, index):
        if len(self.data_sample) and len(self.labels_sample):
            img_path = self.imgs_path[self.data_sample[index]]
            target = self.labels_sample[index]
        else:
            img_path, target = self.imgs_path[index], self.targets[index]

        if self._print:
            self._print("Img Index: {}".format(index, img_path))

        img = default_loader(img_path)
        if self.transform:
            img = self.transform(img)

        return img, target, img_path

    def __len__(self):
        length = 0
        if len(self.data_sample) and len(self.labels_sample):
            length = len(self.labels_sample)
        else:
            length = len(self.targets)
        return length

    def set_data(self, class_index: list, num_classes: int):
        """Sample data equally.
        """
        self.data_sample = []
        self.labels_sample = []

        for index in class_index:
            self.data_sample.extend(self.targets_imgs_dict
                                    [index][: num_classes])
            self.labels_sample.extend([index] * num_classes)
        print("Len of new dataset is :{}".format(len(self.data_sample)))
        self.data_sample = np.array(self.data_sample)
        self.labels_sample = np.array(self.labels_sample)

    @property
    def split_folder(self):
        if self.is_train:
            split = "train"
        else:
            split = "val"
        return os.path.join(self.root, split)

    def read_path(self):
        """Read_path.

            imgs (list): List of (image path, class_index) tuples
        """
        return list(zip(*self.imgs))[:]


def load_meta_file(path):
    """Load meta bin.
    """
    return torch.load(path)


def print_dataset(dataset, print_time):
    """Print dataset.
    """
    print(len(dataset))
    from collections import Counter
    counter = Counter()
    labels = []
    for index, (img, label, img_path) in enumerate(dataset):
        if index % print_time == 0:
            print(img.size(), label, img_path)
        labels.append(label)
    counter.update(labels)
    print("List has {} unique keys".format(len(list(counter))))
    print(counter)


def plot_dataset(targets, save_path):
    """Plot distribution of dataset.
    """
    sns.set()
    fig = sns.distplot(targets).get_figure()
    fig.savefig(save_path)


def test_ImageNet():
    """Test ImageNet.
    """
    # ImageNet
    root = "/media/lincolnzjx/HardDisk/Datasets"
    ds = ImageNet(root=root, is_train=True,
                  transform=transforms.ToTensor())
    print_dataset(ds, 50)
    # ds.set_data([1, 2, 3], 100)
    # print_dataset(ds, 50)
    # ds = ImageNet(root=root, is_train=True,
    #               transform=transforms.ToTensor())
    # ds.set_data([1, 2, 3], 100)
    # print_dataset(ds, 50)
    # print("-" * 100)
    # print_dataset(ds, 1000)
    # for data, target, img_path in ds:
    #     print(data.size(), target, img_path)

    # for index in range(len(ds)):
    #     img, label = ds[index]
    #     print(img.shape, label)

    #  Test set_data, obtaining specific images from specific classes
    #  results: 1,2,3 per 100 images
    # ds = ImageNet(root=root, is_train=False,
    #               transform=transforms.ToTensor())
    # ds.set_data([1], 100)
    # print_dataset(ds, 10)
    # print("-" * 100)
    # targets = ds.targets
    # save_path = os.path.join(root, "ilsvrc2012/distribution.pdf")
    # plot_dataset(targets, save_path)


if __name__ == "__main__":
    test_ImageNet()
