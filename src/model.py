import sys
import torch
import torch.nn as nn
from torchsummary import summary
import copy

from models.GuidedReLu import GuidedBackpropReLU
from models.VGG import VGG16
from models.ResNet import ResNet18
from utils.initialization import _kaiming_normal, _xavier_normal, \
        _kaiming_uniform, _xavier_uniform


class Network(nn.Module):
    """Network
    """
    def __init__(self, backbone="alxenet", num_classes=10, input_channel=1,
                 pretrained=False, dropout=True, conv_bias=True,
                 linear_bias=True, guidedReLU=False, selected_layer=None):
        super(Network, self).__init__()
        if backbone == "alexnet":
            model = AlexNet(num_classes, input_channel)
        elif backbone == "convNet":
            model = convNet()
        elif backbone == "resnet18":
            model = ResNet18(num_classes, input_channel, pretrained,
                             selected_layer=selected_layer)
        elif backbone == "vgg16":
            model = VGG16(num_classes, input_channel, pretrained, dropout,
                          conv_bias, linear_bias,
                          selected_layer=selected_layer)
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

    def set_guildedReLU(self, guidedReLU=False):
        if guidedReLU:
            recursive_relu_apply(self.model)
            print("Using GuidedReLu.")

    def hook_forward(self, activation_maps):
        def forward_hook_fn(module, inputs, outputs):
            """Hook forward function.
            """
            outputs_cpu = outputs.detach().clone().cpu().numpy()
            activation_maps.append(outputs_cpu)

        for name, module in self.model.named_modules():
            new_name = name.replace("resnet18.", "")
            new_name = new_name.replace("features.", "")
            if new_name == str(self.selected_layer):
                print("=> Register fhook {}".format(new_name))
                handler = module.register_forward_hook(forward_hook_fn)
                self.fn_handler.append(handler)

    def get_activation_maps(self, inputs, selected_layer):
        self.model.eval()
        self.selected_layer = selected_layer
        self.activation_maps = []
        self.fn_handler = []
        self.hook_forward(self.activation_maps)
        self.model(inputs)
        for handler in self.fn_handler:
            handler.remove()
        self.forward_hook_handler = []
        return self.activation_maps


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


class Identity(nn.Module):
    """Identity path.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


def init_weights(net, method, _print):
    """Initialize weights of the networks.

        weights of conv layers and fully connected layers are both initialzied
        with Xavier algorithm. In particular, set parameters to random values
        uniformly drawn from [-a, a], where a = sqrt(6 * (din + dout)), for
        batch normalization layers, y=1, b=0, all biases initialized to 0
    """
    if method == "kaiming_normal":
        net = _kaiming_normal(net, _print)
    elif method == "kaiming_uniform":
        net = _kaiming_uniform(net, _print)
    elif method == "xavier_uniform":
        net = _xavier_uniform(net, _print)
    elif method == "xavier_normal":
        net = _xavier_normal(net, _print)
    else:
        _print("Init weight: Need legal initialization method")
    return net


def replace_layer(model, keys=None):
    model = copy.deepcopy(model)
    for name, layer in model.named_modules():
        # Replace maxpool with avgpool
        if "AVG" in keys and isinstance(layer, nn.MaxPool2d):
            print("Replace {} with {}".format(name, "avgPool"))
            first_wrap_name, second_wrap_name = name.split(".")
            second_wrap_name = int(second_wrap_name)
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            ceil_mode = layer.ceil_mode
            model._modules[first_wrap_name][second_wrap_name] = \
                nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
                             padding=padding, ceil_mode=ceil_mode)
        if "LEAK" in keys and isinstance(layer, nn.ReLU):
            print("Replace {} with {}".format(name, "LeakReLU"))
            first_wrap_name, second_wrap_name = name.split(".")
            second_wrap_name = int(second_wrap_name)
            model._modules[first_wrap_name][second_wrap_name] = nn.LeakyReLU()
    return model


def recursive_relu_apply(module_top):
    """Recursively apply GuidedBackpropReLU to ReLU.
    """
    for idx, module in module_top._modules.items():
        recursive_relu_apply(module)
        if module.__class__.__name__ == "ReLU":
            module_top._modules[idx] = GuidedBackpropReLU.apply


if __name__ == "__main__":
    # net = Network(backnone="alexnet")
    input_size = (3, 224, 224)
    net = Network(backbone="vgg16", num_classes=200)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.print_model(input_size, device)
