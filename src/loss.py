"""Loss.

Loss function would be defined here

"""

import sys

import torch
import torch.nn as nn
# import torch.nn.functional as F

from utils.regularization import all2zero, l1_regularization, \
        l2_regularization, total_variation_v2


class FileterLoss(nn.Module):
    """FilterLoss.

    Args:
        model:
        selected_layer(int):
        selected_filter(int):
        mode(str): mode for loss, keep is to keep the feature map for
                   selected filter from selected layer, while remove is
                   to remove the selected filter.
    """
    def __init__(self, model, selected_layer, selected_filter, mode="keep",
                 inter=False, rho=0, regularization="None", p=None,
                 smoothing=False, _print=None):
        super(FileterLoss, self).__init__()
        self.model = model.model.features
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.mode = mode
        self.conv_output = None
        # Generate a random image
        self.hook_layer()
        self.rho = rho
        # Use to interact different channels
        self.inter = inter
        self._print = _print
        # Regularization
        self.regularize_op = self.get_regularize(regularization)
        self.smothing_op = self.get_smoothing(smoothing)
        self.p = p

    def hook_layer(self):
        """Hook layer.
        """
        def hook_function(module, grad_in, grad_out):
            """Hook function.
            """
            # Get the conv output of selected filter (from selected layer)
            # self.conv_output[self.selected_layer] = \
            #                               grad_out[0, self.selected_filter]
            self.conv_output = grad_out

        # Hook the selected layer
        self.model[self.selected_layer].\
            register_forward_hook(hook_function)

    def forward(self, processed_inputs, original_inputs):
        processed_outputs = processed_inputs
        for index, layer in enumerate(self.model):
            processed_outputs = layer(processed_outputs)
            if index == self.selected_layer:
                break
        # Loss function is the mean of the output of selected layer/filter
        # The final loss would be designed at a pixel level
        # Try to minimize the mean of the output of the specific filter

        # Concat tensor along the channels
        rest_processed_feature_map = \
            torch.cat((self.conv_output[:, :self.selected_filter],
                       self.conv_output[:, self.selected_filter+1:]),
                      dim=1)
        # print(rest_processed_feature_map.size())
        selected_processed_feature_map = \
            self.conv_output[:, self.selected_filter]
        # print("selected_processed_feature_map:",
        # selected_processed_feature_map.size())
        *_, height, width = selected_processed_feature_map.size()
        # pixels = height * width

        # Obtain original tensors
        original_outputs = original_inputs
        for index, layer in enumerate(self.model):
            original_outputs = layer(original_outputs)
            if index == self.selected_layer:
                break

        selected_original_feature_map = \
            self.conv_output[:, self.selected_filter]
        # Whether to use interact to ignore some feature map
        if self.inter:
            self._print("Inter")
            rest_processed_feature_map = \
                self.interact(selected_original_feature_map,
                              rest_processed_feature_map)

        if self.mode == "keep":
            selected_filter_loss = torch.norm((selected_original_feature_map -
                                               selected_processed_feature_map),
                                              dim=(1, 2), p=2)
            selected_filter_loss = torch.mean(selected_filter_loss)
            rest_feature_map_norm = torch.norm(rest_processed_feature_map,
                                               dim=(2, 3), p=1)
            rest_filter_loss = torch.mean(rest_feature_map_norm)
        elif self.mode == "remove":
            selected_feature_map_norm = \
                torch.norm(selected_processed_feature_map, dim=(1, 2), p=1)
            selected_filter_loss = torch.mean(selected_feature_map_norm)
            rest_original_feature_map = \
                torch.cat((self.conv_output[:, :self.selected_filter],
                           self.conv_output[:, self.selected_filter+1:]),
                          dim=1)
            rest_filter_loss = torch.norm((rest_original_feature_map -
                                          rest_processed_feature_map),
                                          dim=(2, 3), p=2)
            rest_filter_loss = torch.mean(rest_filter_loss)
        else:
            print("No loss function of mode available")
            sys.exit(-1)

        regularization_loss = self.regularize_op(processed_inputs, p=self.p)
        smoothing_loss = self.smothing_op(processed_outputs, p=self.p)

        return selected_filter_loss, rest_filter_loss, regularization_loss, \
            smoothing_loss

    def interact(self, selected_feature_map, rest_processed_feature_map):
        """Interact between different channels.
        Args:
            selected_feature_map: [batch_size, height, width]
            rest_processed_feature_map: [batch_size, channels, height, width]
        Returns:
            rest_processed_feature_map: [batc_size, channels, height, width]
        """
        (batch_size, channels, height, width) = \
            rest_processed_feature_map.size()
        selected_feature_map_rp = selected_feature_map.unsqueeze(1)
        selected_feature_map_rp = \
            selected_feature_map_rp.repeat(1, channels, 1, 1)

        # Element for selected_feature_map less than rho would be set to 1
        # And used to supress different channels
        selected_feature_map_rp_mask = selected_feature_map_rp.le(self.rho)
        selected_feature_map_rp_mask = \
            selected_feature_map_rp_mask.to(selected_feature_map_rp.dtype)
        rest_processed_feature_map = \
            rest_processed_feature_map * selected_feature_map_rp_mask
        return rest_processed_feature_map

    def get_regularize(self, regularization):
        """Regularize using different strategy.
        """
        regularize_op = None
        if regularization == "None":
            regularize_op = all2zero
        elif regularization == "L1":
            regularize_op = l1_regularization
        elif regularization == "L2":
            regularize_op = l2_regularization
        elif regularization == "ClipNorm":
            pass
        elif regularization == "ClipContribution":
            pass
        else:
            self._print("Need legal regularization method!")
            sys.exit(-1)
        return regularize_op

    def get_smoothing(self, smoothing):
        """Smoothing op.
        """
        smoothing_op = all2zero
        if smoothing == "TotalVariation":
            smoothing_op = total_variation_v2
        else:
            self._print("Need legal smoothing method!")
            sys.exit(-1)
        return smoothing_op


def test_keep():
    """Test keep mode in loss function.
    """
    print("Test keep ----------")
    import model
    convnet = model.Network(backbone="vgg16", pretrained=True)
    filter_loss = FileterLoss(convnet, 1, 0, mode="keep")
    # inputs = torch.rand(3, 3, 224, 224)
    # inputs_processed = torch.rand(3, 3, 224, 224, requires_grad=True)
    inputs = torch.zeros(3, 3, 224, 224)
    inputs_processed = torch.ones(3, 3, 224, 224, requires_grad=True)
    loss = filter_loss(inputs_processed, inputs)
    print(loss)


def test_remove():
    """Test remove mode in loss function.
    """
    print("Test remove ----------")
    import model
    convnet = model.Network(backbone="vgg16", pretrained=True)
    filter_loss = FileterLoss(convnet, 1, 17, mode="remove")
    inputs = torch.zeros(3, 3, 224, 224)
    inputs_processed = torch.ones(3, 3, 224, 224, requires_grad=True)
    loss = filter_loss(inputs_processed, inputs)
    print(loss)


def test_interact():
    """Test interact betweens different chanels.
    """
    print("Test keep ----------")
    rho = 0.1
    print("rho is :{}".format(rho))
    import model
    convnet = model.Network(backbone="vgg16", pretrained=True)
    filter_loss = FileterLoss(convnet, 20, 49, mode="keep", inter=True,
                              rho=rho)
    inputs = torch.zeros(3, 3, 224, 224)
    inputs_processed = torch.ones(3, 3, 224, 224, requires_grad=True)
    loss = filter_loss(inputs_processed, inputs)
    print(loss)
    print("----")
    convnet = model.Network(backbone="vgg16", pretrained=True)
    filter_loss = FileterLoss(convnet, 20, 49, mode="keep", inter=False)
    inputs = torch.zeros(3, 3, 224, 224)
    inputs_processed = torch.ones(3, 3, 224, 224, requires_grad=True)
    loss = filter_loss(inputs_processed, inputs)
    print(loss)


def test_regularize():
    """Test keep mode in loss function.
    """
    import model
    regularization = "None"
    print("Test keep with regularization {} ----------".format(regularization))
    convnet = model.Network(backbone="vgg16", pretrained=True)
    filter_loss = FileterLoss(convnet, 20, 49, mode="keep",
                              regularization=regularization)
    inputs = torch.zeros(3, 3, 224, 224)
    inputs_processed = torch.ones(3, 3, 224, 224, requires_grad=True)
    loss = filter_loss(inputs_processed, inputs)
    print(loss)

    regularization = "L1"
    print("Test keep with regularization {} ----------".format(regularization))
    convnet = model.Network(backbone="vgg16", pretrained=True)
    filter_loss = FileterLoss(convnet, 20, 49, mode="keep",
                              regularization=regularization)
    inputs = torch.zeros(3, 3, 224, 224)
    inputs_processed = torch.ones(3, 3, 224, 224, requires_grad=True)
    loss = filter_loss(inputs_processed, inputs)
    print(loss)


if __name__ == "__main__":
    # test keep
    # test_keep()
    # test remove
    # test_remove()

    # test interact
    # test_interact()

    # test regularization
    test_regularize()
