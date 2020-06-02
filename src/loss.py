"""Loss.

Loss function would be defined here

"""

import sys

import torch
import torch.nn as nn

from utils.regularization import all2zero, l1_regularization, \
        l2_regularization, total_variation_v2
from utils.function import format_print


class FilterLoss(nn.Module):
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
                 smoothing="None", _print=None, defensed=False):
        super(FilterLoss, self).__init__()
        self.model = model.model
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.mode = mode
        # self.conv_output = None
        # Generate a random image
        # self.hook_layer()
        self.rho = rho
        # Use to interact different channels
        self.inter = inter
        self._print = _print
        # Regularization
        self.regularize_op = self.get_regularize(regularization)
        self.smothing_op = self.get_smoothing(smoothing)
        self.p = p
        # Activations.
        self.original_activation_maps = []
        self.processed_activation_maps = []
        self.forward_hook_handler = []
        # self.backward_hook_handler = []

        self.defensed = defensed
        if self.defensed:
            self.hook_backward()

    # def hook_layer(self):
    #     """Hook layer.
    #     """
    #     def hook_function(module, grad_in, grad_out):
    #         """Hook function.
    #         """
    #         # Get the conv output of selected filter (from selected layer)
    #         # self.conv_output[self.selected_layer] = \
    #         #                               grad_out[0, self.selected_filter]
    #         self.conv_output = grad_out

    #     # Hook the selected layer
    #     self.model[self.selected_layer].\
    #         register_forward_hook(hook_function)

    def hook_forward(self, activation_maps):
        def forward_hook_fn(module, inputs, outputs):
            """Hook forward fuction.
            """
            activation_maps.append(outputs)

        if self.defensed:
            for name, module in self.model.named_children():
                if isinstance(module, nn.ReLU):
                    self._print("=> Register fhook {}".format(name))
                    handler = module.register_forward_hook(forward_hook_fn)
                    self.forward_hook_handler.append(handler)

        for name, module in self.named_modules():
            new_name = name.replace("model.resnet18.", "")
            new_name = new_name.replace("model.features.", "")
            if new_name == str(self.selected_layer):
                self._print("=> Register fhook {}".format(new_name))
                handler = module.register_forward_hook(forward_hook_fn)
                self.forward_hook_handler.append(handler)

    def remove_hook_forward(self):
        """Remove forward hook.
        """
        def remove_forward_hook_fn():
            for handler in self.forward_hook_handler:
                handler.remove()
                self._print("=> Remove fhook {}".format(handler))
            self.forward_hook_handler = []
        remove_forward_hook_fn()

    def hook_backward(self):
        def backward_hook_fn(module, grad_in, grad_out):
            grad_output = grad_out[0]
            o_activation_map = self.original_activation_maps.pop()
            p_activation_map = self.processed_activation_maps.pop()
            positive_mask_1 = (p_activation_map > 0).type_as(grad_output)
            positive_mask_2 = (p_activation_map < o_activation_map).\
                type_as(grad_output)
            dummy_zeros_1 = torch.zeros_like(grad_output)
            dummy_zeros_2 = torch.zeros_like(grad_output)

            # ReLU
            grad_input = torch.addcmul(dummy_zeros_1,
                                       positive_mask_1, grad_output)
            # Guilded.
            grad_input = torch.addcmul(dummy_zeros_2,
                                       positive_mask_2, grad_input)
            return (grad_input,)

        for name, module in self.model.named_children():
            if isinstance(module, nn.ReLU):
                print("=> Register bhook {}".format(name))
                module.register_backward_hook(backward_hook_fn)
                # self.backward_hook_handler.append(handler)

    # def remove_backward_hook_fn(self, handler):
    #     """Remove backward hook.
    #     """
    #     def remove_forward_hook_fn():
    #         for handler in self.forward_hook_handler:
    #             handler.remove()
    #             print("=> Remove fhook {}".format(handler))
    #         self.forward_hook_handler = []

    #     pass

    def forward(self, processed_inputs, original_inputs):
        """
        Args:
            processed_inputs: [batch_size, channels, height, width]
            original_inputs: [batch_size, channels, height, width]
        Reuturns:
            selected_filter_loss:
            rest_filter_loss:
            regularization_loss:
            smoothing_loss:
        """
        self.original_activation_maps = []
        self.processed_activation_maps = []

        self.remove_hook_forward()
        self.hook_forward(self.original_activation_maps)

        # Obtain original tensors
        # original_outputs = original_inputs
        # for index, layer in enumerate(self.model):
        #     original_outputs = layer(original_outputs)
        #     if index == self.selected_layer:
        #         break
        # original_outputs =
        self.model(original_inputs)

        selected_original_feature_map = \
            self.original_activation_maps[0][:, self.selected_filter]
        # original_outputs[:, self.selected_filter]
        # processed_outputs = self.model(processed_inputs)

        # if self.defensed:
        self.remove_hook_forward()
        self.hook_forward(self.processed_activation_maps)
        # processed_outputs =
        self.model(processed_inputs)
        # for index, layer in enumerate(self.model):
        #     processed_outputs = layer(processed_outputs)
        #     if index == self.selected_layer:
        #         break

        # print(self.forward_hook_handler)
        # self.conv_output = processed_outputs

        # Loss function is the mean of the output of selected layer/filter
        # The final loss would be designed at a pixel level
        # Try to minimize the mean of the output of the specific filter

        # Concat tensor along the channels
        rest_processed_feature_map = \
            torch.cat(
                (self.processed_activation_maps[0][:, :self.selected_filter],
                 self.processed_activation_maps[0][:,
                                                   self.selected_filter+1:]),
                dim=1)
        # torch.cat((processed_outputs[:, :self.selected_filter],
        #            processed_outputs[:, self.selected_filter+1:]),
        # print(rest_processed_feature_map.size())
        selected_processed_feature_map = \
            self.processed_activation_maps[0][:, self.selected_filter]
        # processed_outputs[:, self.selected_filter]
        # print("selected_processed_feature_map:",
        # selected_processed_feature_map.size())
        *_, height, width = selected_processed_feature_map.size()
        # pixels = height * width

        # Whether to use interact to ignore some feature map
        # TODO, need to return two rest
        rest_processed_feature_map_interact = \
            self.interact(selected_original_feature_map,
                          rest_processed_feature_map)

        if self.mode == "keep":
            selected_filter_norm = torch.norm((selected_original_feature_map -
                                               selected_processed_feature_map),
                                              dim=(1, 2), p=2)
            selected_filter_square = selected_filter_norm ** 2
            selected_filter_square_avg = selected_filter_square /\
                (height * width)
            selected_filter_loss = torch.mean(selected_filter_square_avg)

            rest_feature_map_norm = torch.norm(rest_processed_feature_map,
                                               dim=(2, 3), p=1)
            rest_feature_map_norm_avg = rest_feature_map_norm /\
                (height * width)
            rest_filter_loss = torch.mean(rest_feature_map_norm_avg)
            # intetact
            rest_feature_map_norm_interact = torch.norm(
                rest_processed_feature_map_interact, dim=(2, 3), p=1)
            rest_feature_map_norm_avg_interact = \
                rest_feature_map_norm_interact / (height * width)
            rest_filter_loss_interact = \
                torch.mean(rest_feature_map_norm_avg_interact)
        # elif self.mode == "remove":
        #     rest_original_feature_map = \
        #         torch.cat((original_outputs[:, :self.selected_filter],
        #                    original_outputs[:, self.selected_filter+1:]),
        #                   dim=1)
        #     selected_filter_norm = torch.norm((rest_processed_feature_map -
        #                                        rest_original_feature_map),
        #                                       dim=(2, 3), p=2)
        #     selected_filter_square = selected_filter_norm ** 2
        #     selected_filter_square_avg = selected_filter_square /\
        #         (height * width)
        #     selected_filter_loss = torch.mean(selected_filter_square_avg)

        #    rest_feature_map_norm = torch.norm(selected_processed_feature_map,
        #                                       dim=(1, 2), p=1)
        #    rest_feature_map_norm_avg = rest_feature_map_norm/(height * width)
        #    rest_filter_loss = torch.mean(rest_feature_map_norm_avg)
        else:
            print("No loss function of mode available")
            sys.exit(-1)

        regularization_loss = self.regularize_op(processed_inputs, p=self.p)
        # smoothing_loss = self.smothing_op(processed_inputs, p=self.p)
        # TODO NOT CONSIDER NOW!!
        smoothing_loss = 0.0

        return selected_filter_loss, rest_filter_loss, regularization_loss, \
            smoothing_loss, rest_filter_loss_interact

    def interact(self, selected_feature_map, rest_processed_feature_map):
        """Interact between different channels.
        Args:
            selected_feature_map: [batch_size, height, width]
            rest_processed_feature_map: [batch_size, channels, height, width]
        Returns:
            rest_processed_feature_map: [batc_size, channels, height, width]
        """
        if self.inter:
            self._print("Inter")
        (batch_size, channels, height, width) = \
            rest_processed_feature_map.size()
        selected_feature_map_rp = selected_feature_map.clone().unsqueeze(1)
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
        if smoothing == "None":
            smoothing_op = all2zero
        elif smoothing == "TotalVariation":
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
    name_list = ["selected_filter_loss", "rest_filter_loss",
                 "l1_regularization", "smoothing_loss"]

    selected = [
        [1, 47]
        # [3, 20],
        # [6, 19],
        # [8, 99],
        # [11, 75],
        # [13, 112],
        # [15, 148]
    ]

    for selected_layer, selected_filter in selected:
        convnet = model.Network(backbone="vgg16", pretrained=True,
                                selected_layer=selected_layer)
        print(convnet)
        filter_loss = FilterLoss(convnet, selected_layer, selected_filter,
                                 mode="keep", regularization="L1",
                                 _print=print, defensed=True)
        inputs = torch.zeros(3, 3, 224, 224)
        inputs_processed = torch.ones(3, 3, 224, 224, requires_grad=True)
        loss = filter_loss(inputs_processed, inputs)
        print("selected_layer:{}, selected_filter:{}".format(selected_layer,
                                                             selected_filter))
        format_print(loss, name_list=name_list)
        # print(loss)


def test_remove():
    """Test remove mode in loss function.
    """
    print("Test remove ----------")
    import model
    convnet = model.Network(backbone="vgg16", pretrained=True)
    filter_loss = FilterLoss(convnet, 1, 17, mode="remove")
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
    filter_loss = FilterLoss(convnet, 20, 49, mode="keep", inter=True,
                             rho=rho)
    inputs = torch.zeros(3, 3, 224, 224)
    inputs_processed = torch.ones(3, 3, 224, 224, requires_grad=True)
    loss = filter_loss(inputs_processed, inputs)
    print(loss)
    print("----")
    convnet = model.Network(backbone="vgg16", pretrained=True)
    filter_loss = FilterLoss(convnet, 20, 49, mode="keep", inter=False)
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
    filter_loss = FilterLoss(convnet, 20, 49, mode="keep",
                             regularization=regularization)
    inputs = torch.zeros(3, 3, 224, 224)
    inputs_processed = torch.ones(3, 3, 224, 224, requires_grad=True)
    loss = filter_loss(inputs_processed, inputs)
    print(loss)

    regularization = "L1"
    print("Test keep with regularization {} ----------".format(regularization))
    convnet = model.Network(backbone="vgg16", pretrained=True)
    filter_loss = FilterLoss(convnet, 20, 49, mode="keep",
                             regularization=regularization)
    inputs = torch.zeros(3, 3, 224, 224)
    inputs_processed = torch.ones(3, 3, 224, 224, requires_grad=True)
    loss = filter_loss(inputs_processed, inputs)
    print(loss)


if __name__ == "__main__":
    # est keep
    test_keep()
    # test remove
    # test_remove()

    # test interact
    # test_interact()

    # test regularization
    # test_regularize()
