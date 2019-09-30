import sys
import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
import torch.nn.functional as F



class Network(nn.Module):
    """Network
    """
    def __init__(self, backbone="alxenet", num_classes=10, input_channel=1,
                 pretrained=False):
        super(Network, self).__init__()
        if backbone == "alexnet":
            model = AlexNet(num_classes, input_channel)
        elif backbone == "convNet":
            model = convNet()
        elif backbone == "vgg16":
            model = VGG16(num_classes, input_channel, pretrained)
        else:
            print("Need model")
            sys.exit(-1)
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)

    def print_model(self, input_size, device):
        """Print model structure
        """
        self.model.to(device)
        summary(self.model, input_size)


class AlexNet(nn.Module):
    """AlexNet
    """
    def __init__(self, num_classes, input_channel):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(
            nn.Linear(4*4*50, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
                )

    def forward(self, x):
        out = self.features(x)
        out = out.view(-1, 4*4*50)
        out = self.fc(out)
        return out


class VGG16(nn.Module):
    """Vgg16.
    """
    def __init__(self, num_classes, input_channel, pretrained=False):
        super(VGG16, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=pretrained)
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.fc = nn.Sequential(
            *list(vgg16.classifier.children())[:-1],
            nn.Linear(4096, num_classes))

    def forward(self, inputs):
        out = self.features(inputs)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def init_weights(net, _print):
    """Initialize weights of the networks.

        weights of conv layers and fully connected layers are both initialzied
        with Xavier algorithm. In particular, set parameters to random values
        uniformly drawn from [-a, a], where a = sqrt(6 * (din + dout)), for
        batch normalization layers, y=1, b=0, all biases initialized to 0
    """
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            _print("Init Conv2d")
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            _print("Init Batch-normalization")
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            _print("Init Linear")
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    return net


if __name__ == "__main__":
    #net = Network(backnone="alexnet")
    input_size = (3, 224, 224)
    net = Network(backbone="vgg16", num_classes=200)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.print_model(input_size, device)
