"""Regularization.
"""

import torch
import torch.nn.functional as F
from function import timethis


def all2zero(inputs, p=None):
    """Return 0 for anything input.
    """
    return torch.tensor(0)


def l1_regularization(inputs, p=None):
    """Regularize inputs with L1.
    """
    loss = torch.mean(torch.norm(inputs, p=1, dim=(1, 2, 3)))
    return loss


def l2_regularization(inputs, p=None):
    """Regularize inputs with L1.
    """
    loss = torch.mean(torch.norm(inputs, p=2, dim=(1, 2, 3)))
    return loss


def gaussian_blur(inputs, p=None):
    """Regularize inputs with Gaussian Blur.
    """
    pass


@timethis
def total_variation_v1(inputs, p=2):
    """Total variantion from <Understandiong deep image representation by
    inverting them>.

    Args:
        inputs: [batch_size, channels, height, width]
        p: norm exponent
    """
    def _loss_per_image(image, height, width, p=2):
        """Compute loss per image.

        Args:
            image: [height, width]
            height:
            width:
        """
        loss = 0
        for row in range(height-1):
            for col in range(width-1):
                pixel = image[row][col]
                down = image[row+1][col]
                right = image[row][col+1]
                loss_per_pixel = (pixel - down) ** 2 + (pixel - right) ** 2
                loss_per_pixel = loss_per_pixel ** (p/2)
                loss += loss_per_pixel
        return loss

    # Brutle implementation
    batch_size, channels, height, width = inputs.size()
    inputs = inputs.view(-1, height, width)
    all_channels = batch_size * channels

    loss = 0
    for channel in range(all_channels):
        loss += _loss_per_image(inputs[channel], height, width, p)
    loss = loss / batch_size
    return loss


@timethis
def total_variation_v2(inputs, p=2):
    """Total variantion from <Understandiong deep image representation by
    inverting them>, revised with conv2d, using matrix computation may be
    faster

    Args:
        inputs: [batch_size, channels, height, width]
        p: norm exponent
    """
    dtype = inputs.dtype
    device = inputs.device
    batch_size, channel, height, width = inputs.size()
    filter_w = torch.tensor([[-1, 1]]).view(1, 1, 1, 2).to(dtype).to(device)
    filter_w = filter_w.repeat(channel, 1, 1, 1)
    filter_h = torch.tensor([[-1], [1]]).view(1, 1, 2, 1).to(dtype).to(device)
    filter_h = filter_h.repeat(channel, 1, 1, 1)
    padding_w = (0, 1)
    padding_h = (1, 0)
    grad_w = F.conv2d(inputs, weight=filter_w, padding=padding_w,
                      groups=channel)
    grad_h = F.conv2d(inputs, weight=filter_h, padding=padding_h,
                      groups=channel)
    # [batch_size, channel, height, width]
    grad_w = grad_w[:, :, :, : width]
    grad_h = grad_h[:, :, :height, :]
    grad = grad_w ** 2 + grad_h ** 2
    grad = torch.pow(grad, p/2).sum(dim=(1, 2, 3))
    loss = torch.mean(grad)
    return loss


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputs = torch.rand(30, 3, 224, 224, device=device)
    # inputs = torch.tensor([[1, 0, 0],
    #                        [0, 1, 1],
    #                        [2, 2, 0.0]])
    # inputs = inputs.view(1, 1, 3, 3)
    print("Inputs size {}".format(inputs.size()))
    loss = total_variation_v1(inputs, p=2)
    print("---------- test total variation")
    print(loss)
    print("---------- test total variation V2")
    loss = total_variation_v2(inputs, p=2)
    print(loss)
