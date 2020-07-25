# copyright 2019 jiaxin zhuang
#
#
# ?? license
# ==============================================================================
"""Dataset.
"""
import os
import pandas as pd
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class MNIST(Dataset):
    """MNIST dataset.
    """
    def __init__(self, root, is_train, transform, _print=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform

        mnist_dataset = torchvision.datasets.MNIST(root=self.root,
                                                   train=self.is_train,
                                                   download=False)

        self.data = mnist_dataset.data.cpu().numpy()
        self.labels = mnist_dataset.targets.cpu().numpy()

        self.classes = list(set(self.labels))
        self.targets_imgs_dict = {}
        label_np = np.array(self.labels)
        for target in self.classes:
            indexes = np.nonzero(label_np == target)[0]
            self.targets_imgs_dict.update({target: indexes})

        self.data_sample = []
        self.labels_sample = []

    def __getitem__(self, index):
        if len(self.data_sample) and len(self.labels_sample):
            img = self.data[self.data_sample[index]]
            label = self.labels_sample[index]
        else:
            img = self.data[index]
            label = self.labels[index]

        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = label.astype("int64")

        img_path = "{}_{}.png".format(label, index)
        return img, label, img_path

    def __len__(self):
        length = 0
        if len(self.data_sample) and len(self.labels_sample):
            length = len(self.labels_sample)
        else:
            length = len(self.labels)
        return length

    def set_data(self, class_index: list, num_classes: int):
        """Sample data equally.
        """
        self.data_sample = []
        self.labels_sample = []

        for index in class_index:
            self.data_sample.extend(self.targets_imgs_dict
                                    [index][:num_classes])
            self.labels_sample.extend([index] * num_classes)
        print("Len of new dataset is :{}".format(len(self.data_sample)))
        self.data_sample = np.array(self.data_sample)
        self.labels_sample = np.array(self.labels_sample)

    # def get_data(self, index):
    #     img = Image.fromarray(img)
    #     if self.transform:
    #         img = self.transform(img)
    #     label = self.labels_sample[index]
    #     label = label.astype("int64")
    #     return img, label


class CUB(Dataset):
    """CUB200-2011.
    """
    def __init__(self, root="../data", is_train=True, transform=None,
                 _print=None):
        self.root = os.path.join(root, "CUB_200_2011")
        self.is_train = is_train
        self.transform = transform

        self.imgs_path, self.targets = self.read_path()
        self.classes = list(set(self.targets))

        self.targets_imgs_dict = {}
        targets_np = np.array(self.targets)
        for target in self.classes:
            indexes = np.nonzero(target == targets_np)[0]
            self.targets_imgs_dict.update({target: indexes})

        self.data_sample = []
        self.labels_sample = []
        self._print = _print

    def __getitem__(self, index):
        if len(self.data_sample) and len(self.labels_sample):
            img_path = self.imgs_path[self.data_sample[index]]
            target = self.labels_sample[index]
        else:
            img_path, target = self.imgs_path[index], self.targets[index]

        # print img_path to log for further analysis
        if self._print:
            self._print("Img index: {} with name: {}".format(index, img_path))

        img = default_loader(img_path)
        if self.transform is not None:
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
                                    [index][:num_classes])
            self.labels_sample.extend([index] * num_classes)
        print("Len of new dataset is :{}".format(len(self.data_sample)))
        self.data_sample = np.array(self.data_sample)
        self.labels_sample = np.array(self.labels_sample)

    def read_path(self):
        """Read img, label and split path.
        """
        img_txt_file_path = os.path.join(self.root, 'images.txt')
        img_txt_file = txt_loader(img_txt_file_path, is_int=False)
        img_name_list = img_txt_file

        label_txt_file_path = os.path.join(self.root, "image_class_labels.txt")
        label_txt_file = txt_loader(label_txt_file_path, is_int=True)
        label_list = list(map(lambda x: x-1, label_txt_file))

        train_test_file_path = os.path.join(self.root, "train_test_split.txt")
        train_test_file = txt_loader(train_test_file_path, is_int=True)
        train_test_list = train_test_file

        if self.is_train:
            train_img_path = [os.path.join(self.root, "images", x)
                              for i, x in zip(train_test_list, img_name_list)
                              if i]
            train_targets = [x for i, x in zip(train_test_list, label_list)
                             if i]
            imgs_path = train_img_path
            targets = train_targets
        else:
            test_img_path = [os.path.join(self.root, "images", x)
                             for i, x in zip(train_test_list, img_name_list)
                             if not i]
            test_targets = [x for i, x in zip(train_test_list, label_list) if
                            not i]
            imgs_path = test_img_path
            targets = test_targets

        return imgs_path, targets


def default_loader(path):
    """Default loader.
    """
    from torchvision import get_image_backend
    loader = None
    if get_image_backend() == "accimage":
        loader = accimage_loader(path)
    else:
        loader = pil_loader(path)
    return loader


def accimage_loader(path):
    """Accimage loader for accelebrating loading image.
    """
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentionally a decoding problem, fall back to PIL.image
        return pil_loader(path)


def pil_loader(path):
    """Image Loader."""
    with open(path, "rb") as afile:
        img = Image.open(afile)
        return img.convert("RGB")


def txt_loader(path, is_int=True):
    """Txt Loader
    Args:
        path:
        is_int: True for labels and split, False for image path
    Returns:
        txt_array: array
    """
    txt_array = []
    with open(path) as afile:
        for line in afile:
            txt = line[:-1].split(" ")[-1]
            if is_int:
                txt = int(txt)
            txt_array.append(txt)
        return txt_array


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


class Caltech101(Dataset):
    """Caltech101 dataset.
    """
    def __init__(self, root="../data/", is_train=True, transform=None,
                 _print=None):
        self.root = os.path.join(root, "Caltech101")
        self.is_train = is_train
        self.transform = transform
        self._print = _print

        self.imgs_path, self.targets = self.read_path()
        self.classes = list(set(self.targets))

        self.targets_imgs_dict = {}
        targets_np = np.array(self.targets)
        for target in self.classes:
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

        # print img_path to log for further analysis
        if self._print:
            self._print("Img index: {} with name: {}".format(index, img_path))

        img = default_loader(img_path)
        if self.transform is not None:
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
                                    [index][:num_classes])
            self.labels_sample.extend([index] * num_classes)
        print("Len of new dataset is :{}".format(len(self.data_sample)))
        self.data_sample = np.array(self.data_sample)
        self.labels_sample = np.array(self.labels_sample)

    def read_path(self):
        """Read all path from csv file.
        """
        # use the first kind of split
        img_file_path = os.path.join(self.root, "split", "csv")

        if self.is_train:
            img_file_path = os.path.join(img_file_path,
                                         "caltech101_train_subset0.csv")
        else:
            img_file_path = os.path.join(img_file_path,
                                         "caltech101_test_subset0.csv")
        csvfile = pd.read_csv(img_file_path, header=None).values
        imgs_path, targets = self.get_imgs_labels_from_csvfile(csvfile)
        return imgs_path, targets

    def get_imgs_labels_from_csvfile(self, csvfile):
        """Find class to index in csvfile. Obtain imgs and labels
        """
        imgs_dir = os.path.join(self.root, "101_ObjectCategories")

        labels = [label[0].split("/")[0] for label in csvfile]
        unique_class = list(set(labels))
        unique_class.sort()
        class_to_index = {unique_class[i]: i for i in range(len(unique_class))}

        # csvfile iteration to generate absolute path and targets
        targets = [class_to_index[label] for label in labels]
        imgs_path = [os.path.join(imgs_dir, path[0]) for path in csvfile]
        return imgs_path, targets


# class ImageNet(Dataset):
#     """ImageNet reader."""


def test_caltech101():
    """Test caltech101.
    """
    # Caltech101
    print("Detail of Train")
    ds = Caltech101(root="../data", is_train=True,
                    transform=transforms.ToTensor())
    print_dataset(ds, 1000)

    print("Detail of Val")
    ds = Caltech101(root="../data", is_train=False,
                    transform=transforms.ToTensor())
    print_dataset(ds, 1000)


def test_cub():
    """Test cub.
    """
    # CUB200
    ds = CUB(root="../data", is_train=True, transform=transforms.ToTensor())
    print_dataset(ds, 1000)
    # for data, target in ds:
    #     print(data.size(), target)

    ds = CUB(root="../data", is_train=False, transform=transforms.ToTensor())
    print_dataset(ds, 1000)
    # for data, target in ds:
    #     print(data.size(), target)


def test_mnist():
    """test mnist.
    """
    # MNIST
    ds = MNIST(root="../data", is_train=True, transform=transforms.ToTensor())
    # for data, target in ds:
    #     print(data.size(), target)

    # for index in range(len(ds)):
    #     img, label = ds[index]
    #     print(img.shape, label)

    #  Test set_data, obtaining specific images from specific classes
    #  results: 1,2,3 per 100 images
    ds.set_data([1, 2, 3], 100)
    print_dataset(ds, 50)
    # results: 1 per 100 images
    ds.set_data([1], 100)
    print_dataset(ds, 50)


# TODO
def test_ImageNet():
    """Test ImageNet.
    """
    pass


if __name__ == "__main__":
    test_caltech101()
    # test_cub()
    # test_mnist()
