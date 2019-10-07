# copyright 2019 jiaxin zhuang
#
#
# ?? license
# ==============================================================================
"""Loss.

Loss function would be defined here

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, model, selected_layer, selected_filter, mode="keep"):
        super(FileterLoss, self).__init__()
        self.model = model.model.features
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.mode = mode
        self.conv_output = None
        # Generate a random image
        self.hook_layer()

    def hook_layer(self):
        """Hook layer.
        """
        def hook_function(module, grad_in, grad_out):
            """Hook function.
            """
            # Get the conv output of selected filter (from selected layer)
            #self.conv_output[self.selected_layer] = grad_out[0, self.selected_filter]
            self.conv_output = grad_out

        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def forward(self, processed_inputs, original_inputs):
        processed_outputs = processed_inputs
        for index, layer in enumerate(self.model):
            processed_outputs = layer(processed_outputs)
            if index == self.selected_layer:
                break
        # Loss function is the mean of the output of selected layer/filter
        # Try to minimize the mean of the output of the specific filter

        #print(self.conv_output[:, :self.selected_filter].size())
        #print(self.conv_output[:, self.selected_filter+1: ].size())

        # Concat tensor along the channels
        rest_processed_feature_map = torch.cat((self.conv_output[:, :self.selected_filter],
                                                self.conv_output[:, self.selected_filter+1:]),
                                               dim=1)
        #print(rest_processed_feature_map.size())
        selected_processed_feature_map = self.conv_output[:, self.selected_filter]
        #print("selected_processed_feature_map:", selected_processed_feature_map.size())

        # Obtain original tensors
        original_outputs = original_inputs
        for index, layer in enumerate(self.model):
            original_outputs = layer(original_outputs)
            if index == self.selected_layer:
                break

        if self.mode == "keep":
            selected_original_feature_map = self.conv_output[:, self.selected_filter]
            selected_filter_loss = F.mse_loss(selected_processed_feature_map,
                                              selected_original_feature_map)
            #print("rest_feature_map:", rest_feature_map.size())
            rest_feature_map_norm = torch.norm(rest_processed_feature_map, dim=(2, 3))
            #print("rest_feature_map_norm:", rest_feature_map_norm.size())
            rest_filter_loss = torch.mean(rest_feature_map_norm)
        elif self.mode == "remove":
            selected_feature_map_norm = torch.norm(selected_processed_feature_map, dim=(1, 2))
            selected_filter_loss = torch.mean(selected_feature_map_norm)
            rest_original_feature_map = torch.cat((self.conv_output[:, :self.selected_filter],
                                                   self.conv_output[:, self.selected_filter+1:]),
                                                  dim=1)
            rest_filter_loss = F.mse_loss(rest_original_feature_map,
                                          rest_processed_feature_map)
        return selected_filter_loss, rest_filter_loss


def test_keep():
    """Test keep mode in loss function.
    """
    print("Test keep ----------")
    import model
    convnet = model.Network(backbone="vgg16")
    filter_loss = FileterLoss(convnet, 0, 17, mode="keep")
    inputs = torch.rand(3, 3, 224, 224)
    inputs_processed = inputs.clone().detach().requires_grad_(True)
    loss = filter_loss(inputs_processed, inputs)
    print(loss)


def test_remove():
    """Test remove mode in loss function.
    """
    print("Test remove ----------")
    import model
    convnet = model.Network(backbone="vgg16")
    filter_loss = FileterLoss(convnet, 0, 17, mode="remove")
    inputs = torch.rand(3, 3, 224, 224)
    inputs_processed = inputs.clone().detach().requires_grad_(True)
    loss = filter_loss(inputs_processed, inputs)
    print(loss)

if __name__ == "__main__":
    # test keep
    test_keep()
    # test remove
    test_remove()
