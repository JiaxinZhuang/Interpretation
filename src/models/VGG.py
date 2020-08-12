"""VGG.
"""
import torch.nn as nn
import torchvision


class VGG16(nn.Module):
    """Vgg16.
    """
    def __init__(self, num_classes, input_channel, pretrained=False,
                 dropout=True, conv_bias=True, linear_bias=True,
                 selected_layer=None):
        """INit.
            if num_classes is 1000, use all original model.
        """
        super(VGG16, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=pretrained)

        if selected_layer is not None:
            self.selected_layer = int(selected_layer)
        else:
            self.selected_layer = None

        # Whether to remove some layers.
        if selected_layer is not None:
            print("Only Keep {}th layers before.".format(selected_layer))
            new_model = nn.Sequential(
                *list(vgg16.features.children())[: self.selected_layer+1]
            )
            self.features = new_model
            self.avgpool = None
            self.fc = None
        else:
            print("Entire model.")
            self.features = vgg16.features
            self.avgpool = vgg16.avgpool
            if num_classes == 1000:
                self.fc = vgg16.classifier
            else:
                self.fc = nn.Sequential(
                    *list(vgg16.classifier.children())[:-1],
                    nn.Linear(4096, num_classes))

        if input_channel == 1:
            print("Replace input channel with 1.")
            self.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3),
                                         stride=(1, 1), padding=(1, 1))
        # if not dropout:
        #     self.remove_dropout()
        # if not conv_bias:
        #     self.remove_conv_bias()
        # if not linear_bias:
        #     self.remove_linear_bias()

    def forward(self, inputs):
        out = self.features(inputs)
        if self.selected_layer is None:
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        return out

    # def remove_dropout(self):
    #     """Replace dropout using direct connection.
    #     """
    #     for name, children in self.fc.named_children():
    #         if isinstance(children, nn.Dropout):
    #             self.fc[int(name)] = Identity()

    # def remove_conv_bias(self):
    #     """Remove bias for conv2d.
    #     """
    #     for name, param in self.features.named_children():
    #         if "bias" in name:
    #             param.bias.fill_(0)
    #             param.requires_grad_(False)

    # def remove_linear_bias(self):
    #     """Remove bias for linear.
    #     """
    #     for name, param in self.classifier.named_children():
    #         if "bias" in name:
    #             param.bias.fill_(0)
    #             param.requires_grad_(False)


class VGG11(nn.Module):
    """Vgg11.
    """
    def __init__(self, num_classes, input_channel, pretrained=False,
                 dropout=True, conv_bias=True, linear_bias=True,
                 selected_layer=None):
        super(VGG11, self).__init__()
        vgg11 = torchvision.models.vgg11(pretrained=pretrained)
        self.selected_layer = None

        # Whether to remove some layers.
        if selected_layer is not None:
            self.selected_layer = int(selected_layer)
            print("Only Keep {}th layers before.".format(selected_layer))
            new_model = nn.Sequential(
                *list(vgg11.features.children())[: self.selected_layer+1]
            )
            self.features = new_model
            self.avgpool = None
            self.fc = None
        else:
            print("Entire model.")
            self.features = vgg11.features
            self.avgpool = vgg11.avgpool
            if num_classes == 1000:
                self.fc = vgg11.classifier
            else:
                self.fc = nn.Sequential(
                    *list(vgg11.classifier.children())[:-1],
                    nn.Linear(4096, num_classes))

        self.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1))
        # if not dropout:
        #     self.remove_dropout()
        # if not conv_bias:
        #     self.remove_conv_bias()
        # if not linear_bias:
        #     self.remove_linear_bias()

    def forward(self, inputs):
        out = self.features(inputs)
        if self.selected_layer is None:
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        return out
