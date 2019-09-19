# Jiaxin Zhuang
#

import torch

class DeepDreamLoss(nn.Module):
    """DeepDreamLoss.
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Generate a random image
        #self.created_image = Image.open(im)
        self.hook_layer()

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Get the conv output of selected filter (from selected layer)
            self.model[self.selected_layer] = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def forward(self, inputs):
        out = inputs
        for index, layer in enumerate(self.model):
            out = layer(out)
            if index == self.selected_layer:
                break
        # Loss function is the mean of the output of selected layer/filter
        # Try to minimize the mean of the output of the specific filter
        loss = -torch.mean(self.conv_output)
        return loss
