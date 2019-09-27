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



if __name__ == "__main__":
    ds = MNIST(root="../data", is_train=True, transform=transforms.ToTensor())
    #for data, target in ds:
    #    print(data.size(), target)
    ds.set_data([1,2,3], 100)

    for index in range(120):
        img, label = ds.get_data(index)
        print(img.shape, label)

    #ds = MNIST(root="../data", is_train=False, transform=transforms.ToTensor())
    #for data, target in ds:
    #    print(data.size(), target)
