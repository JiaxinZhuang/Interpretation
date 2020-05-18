"""DTD datasets.
"""


import os
import torchvision
from torchvision.datasets.folder import default_loader


class DTD(torchvision.datasets.ImageFolder):
    """DTD Dataset.
    """
    def __init__(self, root="../data", is_train=True, transform=None,
                 _print=None):
        self.root = os.path.join(root, "dtd", "images")
        super(DTD, self).__init__(root=root, transform=transform)
        self.is_train = is_train
        self.transform = transform
        self._print = _print

    def __getitem__(self, index):
        img_path, target = self.samples
        img = default_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target
