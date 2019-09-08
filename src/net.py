import torch.nn as nn
import torchvision
from torchsummary import summary


class Network(nn.Module):
    """Network
    """
    def __init__(self, backnone="alxenet"):
        super(Network, self).__init__()
        if backbone == "alexnet":
            model = torchvision.models.alexnet()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)

    def print_model(self, input_size):
        """Print model structure
        """
        summary(self.model.cuda(), input_size)


if __name__ == "__main__":
    net = Network(backnone="alexnet")
    net.print_model()