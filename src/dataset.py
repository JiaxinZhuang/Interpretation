import os
import torchvision
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms


class MNIST(Dataset):
    def __init__(self, root, is_train, transform, input_size=224):
        self.root = root
        self.is_train = is_train
        self.transform = transform

        mnist_dataset = torchvision.datasets.MNIST(root=self.root, train=self.is_train, download=False)

        self.data = mnist_dataset.data.cpu().numpy()
        self.labels = mnist_dataset.targets.cpu().numpy()

        self.classes = list(set(self.labels))
        self.target_img_dict = dict()
        label_np = np.array(self.labels)
        for target in self.classes:
            indexes = np.nonzero(label_np == target)[0]
            self.target_img_dict.update({target: indexes})

        self.data_sample = []
        self.labels_sample = []

    def __getitem__(self, index):
        img = self.data[self.data_sample[index]]
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        label = label.astype("int64")
        return img, label

    def __len__(self):
        return len(self.labels)

    def set_data(self, class_index: list, num_classes: int):
        """Sample data equally.
        """
        self.data_sample = []
        self.labels_sample = []

        for index in class_index:
            self.data_sample.extend(self.target_img_dict[index][:num_classes])
            self.labels_sample.extend([index] * num_classes)
        print("Len of new dataset is :{}".format(len(self.data_sample)))
        self.data_sample = np.array(self.data_sample)
        self.labels_sample = np.array(self.labels_sample)

    #def get_data(self, index):
    #    img = Image.fromarray(img)
    #    if self.transform:
    #        img = self.transform(img)
    #    label = self.labels_sample[index]
    #    label = label.astype("int64")
    #    return img, label


class CUB(Dataset):
    """CUB200-2011.
    """
    def __init__(self, root="../data", is_train=True, transform=None):
        self.root = os.path.join(root, "CUB_200_2011")
        self.is_train = is_train
        self.transform = transform

        self.imgs_path, self.targets = self.read_path()
        self.classes = list(set(self.targets))

        self.targets_imgs_dict = dict()
        targets_np = np.array(self.targets)
        for target in self.classes:
            indexes = np.nonzero(target == targets_np)[0]
            self.targets_imgs_dict.update({target: indexes})

    def __getitem__(self, index):
        img_path, target = self.imgs_path[index], self.targets[index]
        img = default_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.targets)

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
            train_img_path = [os.path.join(self.root, "images", x) \
                              for i, x in zip(train_test_list, img_name_list) if i]
            train_targets = [x for i, x in zip(train_test_list, label_list) if i]
            imgs_path = train_img_path
            targets = train_targets
        else:
            test_img_path = [os.path.join(self.root, "images", x) \
                             for i, x in zip(train_test_list, img_name_list) if not i]
            test_targets = [x for i, x in zip(train_test_list, label_list) if not i]
            imgs_path = test_img_path
            targets = test_targets

        return imgs_path, targets

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

def accimage_loader(path):
    import accimage
    try:
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
    print(len(dataset))
    from collections import Counter
    counter = Counter()
    labels = []
    for index, (img, label) in enumerate(dataset):
        if index % print_time == 0:
            print(img.size(), label)
        labels.append(label)
    counter.update(labels)
    print(counter)



if __name__ == "__main__":
    # CUB200
    ds = CUB(root="../data", is_train=True, transform=transforms.ToTensor())
    print_dataset(ds, 1000)
    #for data, target in ds:
    #    print(data.size(), target)

    ds = CUB(root="../data", is_train=False, transform=transforms.ToTensor())
    print_dataset(ds, 1000)
    #for data, target in ds:
    #    print(data.size(), target)


    # MNIST
    #ds = MNIST(root="../data", is_train=True, transform=transforms.ToTensor())
    #for data, target in ds:
    #    print(data.size(), target)
    #ds.set_data([1,2,3], 100)

    #for index in range(120):
    #    img, label = ds.get_data(index)
    #    print(img.shape, label)
    #ds = MNIST(root="../data", is_train=False, transform=transforms.ToTensor())
    #for data, target in ds:
    #    print(data.size(), target)
