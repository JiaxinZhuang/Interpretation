"""GuidedBackpropReLU.
"""

import torch
from torch.autograd import Function


class GuidedBackpropReLU(Function):
    """GuidedBackpropReLU operation, which a variant of ReLU.
    Only positive Grad can be propagate back to input.
    """
    @staticmethod
    def forward(self, inputs):
        """Forward.
        ReLU, save input for backward.
        """
        positive_mask = (inputs > 0).type_as(inputs)
        output = torch.addcmul(torch.zeros_like(inputs),
                               positive_mask, inputs)
        self.save_for_backward(inputs)
        return output

    @staticmethod
    def backward(self, grad_output):
        """Backward.
        Only passed by positive grad_output
        """
        (inputs,) = self.saved_tensors
        positive_mask_1 = (inputs > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        dummy_zeros_1 = torch.zeros_like(grad_output)
        dummy_zeros_2 = torch.zeros_like(grad_output)

        # ReLU
        grad_input = torch.addcmul(dummy_zeros_1,
                                   positive_mask_1, grad_output)
        # Guilded.
        grad_input = torch.addcmul(dummy_zeros_2,
                                   positive_mask_2, grad_input)
        return grad_input
# class GuidedBackpropReLU(Function):
#     """GuidedBackpropReLU operation, which a variant of ReLU.
#     Only positive Grad can be propagate back to input.
#     """
#     @staticmethod
#     def forward(self, inputs):
#         """Forward.
#         ReLU, save input for backward.
#         """
#         positive_mask = (inputs > 0).type_as(inputs)
#         output = torch.addcmul(torch.zeros_like(inputs),
#                                positive_mask, inputs)
#         self.save_for_backward(inputs, output)
#         return output
#
#     @staticmethod
#     def backward(self, grad_output):
#         """Backward.
#         Only passed by positive grad_output
#         """
#         inputs, _ = self.saved_tensors
#         positive_mask_1 = (inputs > 0).type_as(grad_output)
#         positive_mask_2 = (grad_output > 0).type_as(grad_output)
#         dummy_zeros_1 = torch.zeros_like(grad_output)
#         dummy_zeros_2 = torch.zeros_like(grad_output)
#
#         # ReLU
#         grad_input = torch.addcmul(dummy_zeros_1,
#                                    positive_mask_1, grad_output)
#         # Guilded.
#         grad_input = torch.addcmul(dummy_zeros_2,
#                                    positive_mask_2, grad_input)
#         return grad_input
