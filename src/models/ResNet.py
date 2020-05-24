"""ResNet.
"""
import torch.nn as nn
import torchvision


class ResNet18(nn.Module):
    """ResNet18."""
    def __init__(self, num_classes, input_channel, pretrained=True,
                 selected_layer=None):
        """INit.
            if num_classes is 1000, use all original model.
        """
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=pretrained)

        self.selected_layer = selected_layer

        # Whether to remove some layers.
        # if selected_layer is not None:
        #     print("Only Keep {}th layers before.".format(selected_layer))
        #     new_model = nn.Sequential(
        #         *list(resnet18.features.children())[: selected_layer+1]
        #     )
        #     self.features = new_model
        #     self.avgpool = None
        #     self.fc = None
        # else:
        #     print("Entire model.")
        #     self.features = vgg16.features
        #     self.avgpool = vgg16.avgpool
        #     if num_classes == 1000:
        #         self.fc = vgg16.classifier
        #     else:
        #         self.fc = nn.Sequential(
        #             *list(vgg16.classifier.children())[:-1],
        #             nn.Linear(4096, num_classes))

    def forward(self, inputs):
        out = self.resnet18(inputs)
        return out
