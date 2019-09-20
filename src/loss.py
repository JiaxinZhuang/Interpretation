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
    """
    def __init__(self, model, selected_layer, selected_filter):
        super(FileterLoss, self).__init__()
        self.model = model.model.features
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
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
        rest_feature_map = torch.cat((self.conv_output[0, :self.selected_filter],
                                      self.conv_output[0, self.selected_filter+1:]),
                                     dim=0)
        #print("rest_feature_map:", rest_feature_map.size())
        rest_feature_map_norm = torch.norm(rest_feature_map, dim=(1,2))
        #print("rest_feature_map_norm:", rest_feature_map_norm.size())
        rest_fileter_loss = torch.mean(rest_feature_map_norm)
        selected_processed_feature_map = self.conv_output[0, self.selected_filter]
        #print("selected_processed_feature_map:", selected_processed_feature_map.size())

        original_outputs = original_inputs
        for index, layer in enumerate(self.model):
            original_outputs = layer(original_outputs)
            if index == self.selected_layer:
                break
        selected_original_feature_map = self.conv_output[0, self.selected_filter]
        selected_filter_loss = F.mse_loss(selected_processed_feature_map,
                                          selected_original_feature_map)
        return selected_filter_loss, rest_fileter_loss


if __name__ == "__main__":
    from torchvision.models import vgg11
    vgg = vgg11(pretrained=False).features
    filterloss = FileterLoss(vgg, 12, 73, 0.5)
    inputs = torch.rand(1, 3, 28, 28)
    inputs_processed = inputs.clone().detach().requires_grad_(True)
    loss = filterloss(inputs_processed, inputs)
    print(loss)
