"""Custom Layers for a Wasserstein Generative Adversarial Network

This module contains custom layers used in the Wasserstein Generative Adversarial Network Architecture.

Classes:
    Generator(nn.Module): Implements a Convolutional generator for GAN which taken in random noise and generates image from it.
    Critic(nn.Module): Implements a Convolutional Critic to criticize the fake or real image from the generator
"""

import torch
from torch import nn


class Generator(nn.Module):
    """
    Generator Module for the Wasserstein GAN.

    This class implements the Wasserstein GAN's generator module. It takes an input tensor of shape
    (batch_size, latent_dim, 1, 1) and applies the generator model to generate an image, The target image basically.
    It converts random noise into (3, 64, 64) sized output image.

    Parameters:
        latent_dim (int): The dimension of the input noise for the generator.
        mid_channels (int): The no of channels to use in mid-layers. We multiply a certain number with this number
        out_channels (int): The no of channels to produce in the output image.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, latent_dim, 1, 1).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, 64, 64).
    """

    def __init__(self, latent_dim: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            # Output Shape: [1, 1024, 2, 2]
            *self.generator_block(latent_dim, mid_channels * 16),
            # Output Shape: [1, 512, 4, 4]
            *self.generator_block(mid_channels * 16, mid_channels * 8),
            # Output Shape: [1, 256, 8, 8]
            *self.generator_block(mid_channels * 8, mid_channels * 4),
            # Output Shape: [1, 128, 16, 16]
            *self.generator_block(mid_channels * 4, mid_channels * 2),
            # Output Shape: [1, 64, 32, 32]
            *self.generator_block(mid_channels * 2, mid_channels),
            # Output Shape: [1, 3, 64, 64]
            nn.ConvTranspose2d(
                mid_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    @staticmethod
    def generator_block(in_channels, out_channels, kernel=4, stride=2, padding=1) -> nn.Sequential:
        """
        Generator Block for the Generator Model.

        This function returns a generator block for the Generator Model. Takes in in_channels, out_channels, kernel, stride, padding
        and returns a Sequential Block with one Transposed Convolution Layer, one normalized layer, and a Relu Activation Function.

        Parameters:
            in_channels (int): The number of channels in the input
            out_channels (int): The number of channels to produce in the output
            kernel (int): The size of kernel (filter)
            stride (int): The stride to use on the input
            padding (int): The padding to apply to the image

        Returns:
            torch.nn.Sequential : Single Generator Block including the defined layers
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass of the Generator.

        This method applies the forward pass of the generator model to the input tensor x.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, latent_dim, 1, 1).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_channels, 64, 64).
        """
        return self.model(x)


class Critic(nn.Module):
    """
    Critic Module for the Wasserstein GAN.

    This class implements the Wasserstein GAN's Critic module. It takes an input tensor of shape
    (batch_size, img_channels, 64, 64) and applies the critic model to criticize the fake and real images.
    The fake ones being generated by the generator.

    Parameters:
        in_channels (int): The number of channels in the input
        mid_channels (int): The number of channels to use in the mid-layers. Again used by multiplying a certain number at each layer
        out_channels (int): The number of channels to produce in the output
    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, 64, 64).

    Returns:
        Either the input image is real or fake.
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            # Output Shape: [1, 64, 32, 32]
            *self.critic_block(in_channels, mid_channels, normalize=False),
            # Output Shape: [1, 128, 16, 16]
            *self.critic_block(mid_channels, mid_channels * 2),
            # Output Shape: [1, 256, 8, 8]
            *self.critic_block(mid_channels * 2, mid_channels * 4),
            # Output Shape: [1, 512, 4, 4]
            *self.critic_block(mid_channels * 4, mid_channels * 8),
            # Output Shape: [1, 1, 1, 1]
            nn.Conv2d(mid_channels * 8, out_channels, 4, 2, 0),
        )

    @staticmethod
    def critic_block(in_channels, out_channels, normalize=True):
        """
        Critic Block for the Generator Model.

        This function returns a critic block for the Critic Model. Takes in in_channels, out_channels,
        and returns a Sequential Block with one Convolution Layer, one normalization layer and a LeakyRelu Activation Function.

        Parameters:
            in_channels (int): The number of channels in the input
            out_channels (int): The number of channels to produce in the output
            normalize (bool): Whether to use the normalization layer or not

        Returns:
            torch.nn.Sequential : Single Critic Block including the defined layers
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(out_channels, affine=True) if normalize else nn.Identity(),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        """
        Forward pass of the Critic.

        This method applies the forward pass of the critic model to the input tensor x.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, img_channels, 64, 64).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, 1, 1).
        """
        return self.model(x)
