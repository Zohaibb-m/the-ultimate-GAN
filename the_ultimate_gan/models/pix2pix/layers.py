"""Custom Layers for a Pix2Pix Generative Adversarial Network

This module contains custom layers used in the Pix2Pix Generative Adversarial Network Architecture.

Classes:
    Generator(nn.Module): Implements a UNET generator for GAN which taken in input image and generates image from it.
    Discriminator(nn.Module): Implements a Convolutional Discriminator(PatchGAN) to detect fake or real image from the generator
"""

import torch
from torch import nn


class UNetDownStep(nn.Module):
    """
    UNET's Down Layer for the Generator.

    This class implements the UNET GAN's Down Step module. It takes in no of input channels, output channels,
    normalize and a dropout value to pass the input through a series of layers
    Parameters:
        in_channels (int): The no of channels in the input image
        out_channels (int): The no of channels to produce in the output image.
        normalize (bool): Whether to use normalization layer or not
        dropout (float): The value for the dropout layer. (No value for not using the dropout)

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

    Returns:
        torch.Tensor: Output tensor with a Convolution applied to it.
    """

    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels) if normalize else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout) if dropout else nn.Identity(),
        )

    def forward(self, x):
        """
        Forward pass of the Generator's Down Sample.

        This method applies the forward pass of the generator's up sample layer to the input tensor x.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, img_channels, H, W).

        Returns:
            torch.Tensor: Output tensor with down convolution applied.
        """
        return self.layer(x)


class UNetUpStep(nn.Module):
    """
    UNET's Up Layer for the Generator.

    This class implements the UNET GAN's Up Step module. It takes in no of input channels, output channels,
    and a dropout value to pass the input through a series of layers
    Parameters:
        in_channels (int): The no of channels in the input image
        out_channels (int): The no of channels to produce in the output image.
        dropout (float): The value for the dropout layer. (No value for not using the dropout)

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

    Returns:
        torch.Tensor: Output tensor with a Convolution applied to it.
    """

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False), nn.InstanceNorm2d(out_channels), nn.Dropout(dropout) if dropout else nn.Identity()
        )

    def forward(self, x, skip_input):
        """
        Forward pass of the Generator's Up Sample.

        This method applies the forward pass of the generator's up sample layer to the input tensor x.

        Parameters:
            skip_input (torch.Tensor): The input from the corresponding Down Step
            x (torch.Tensor): Input tensor of shape (batch_size, img_channels, H, W).

        Returns:
            torch.Tensor: Output tensor with up convolution applied
        """
        x = self.layer(x)
        x = torch.cat([x, skip_input], 1)
        return x


class Generator(nn.Module):
    """
    Generator Module for the Pix2Pix GAN.

    This class implements the Pix2Pix GAN's generator module. It takes an input image of shape
    (batch_size, in_channels, H, W) and applies the generator model to generate an image, The target image basically.

    Parameters:
        in_channels (int): The no of channels in the input image.
        out_channels (int): The no of channels to produce in the output image.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unet_down_layers = nn.Sequential(
            UNetDownStep(in_channels, 64, normalize=False),
            UNetDownStep(64, 128),
            UNetDownStep(128, 256),
            UNetDownStep(256, 512, dropout=0.5),
            UNetDownStep(512, 512, dropout=0.5),
            UNetDownStep(512, 512, dropout=0.5),
            UNetDownStep(512, 512, dropout=0.5),
            UNetDownStep(512, 512, dropout=0.5, normalize=False),
        )

        self.unet_up_layers = nn.Sequential(
            UNetUpStep(512, 512, dropout=0.5),
            UNetUpStep(1024, 512, dropout=0.5),
            UNetUpStep(1024, 512, dropout=0.5),
            UNetUpStep(1024, 512, dropout=0.5),
            UNetUpStep(1024, 256),
            UNetUpStep(512, 128),
            UNetUpStep(256, 64),
        )

        self.final_layer = nn.Sequential(nn.Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(128, out_channels, kernel_size=4, padding=1), nn.Tanh())

    def forward(self, x):
        """
        Forward pass of the Generator.

        This method applies the forward pass of the generator model to the input tensor x.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, img_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, img_channels, H, W).
        """
        dl1 = self.unet_down_layers[0](x)
        dl2 = self.unet_down_layers[1](dl1)
        dl3 = self.unet_down_layers[2](dl2)
        dl4 = self.unet_down_layers[3](dl3)
        dl5 = self.unet_down_layers[4](dl4)
        dl6 = self.unet_down_layers[5](dl5)
        dl7 = self.unet_down_layers[6](dl6)
        dl8 = self.unet_down_layers[7](dl7)

        ul1 = self.unet_up_layers[0](dl8, dl7)
        ul2 = self.unet_up_layers[1](ul1, dl6)
        ul3 = self.unet_up_layers[2](ul2, dl5)
        ul4 = self.unet_up_layers[3](ul3, dl4)
        ul5 = self.unet_up_layers[4](ul4, dl3)
        ul6 = self.unet_up_layers[5](ul5, dl2)
        ul7 = self.unet_up_layers[6](ul6, dl1)
        return self.final_layer(ul7)


class Discriminator(nn.Module):
    """
    Discriminator Module for the Pix2Pix GAN.

    This class implements the Pix2Pix GAN's discriminator module. It takes two input image of shape
    (batch_size, in_channels, H, W) and applies the discriminator model to discrminate the first image as real or fake.

    Parameters:
        in_channels (int): The no of channels in the input image.

    Inputs:
        img1 (torch.Tensor): Input Image of shape (batch_size, in_channels, H, W).
        img2 (torch.Tensor): Input Image of shape (batch_size, in_channels, H, W).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, patch_height, patch_width).
    """

    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            *self.discriminator_block(in_channels * 2, 64, normalize=False),
            *self.discriminator_block(64, 128),
            *self.discriminator_block(128, 256),
            *self.discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, 1, 1, bias=False)
        )

    @staticmethod
    def discriminator_block(in_channels, out_channels, normalize=True):
        """
        Discriminator Block for the Discriminator Model.

        This function returns a discriminator block for the Discriminator Model. Takes in in_channels, out_channels,
        and a normalize boolean and returns a Sequential Block with one Convolution Layer, normalization layer(incase normalize is required)
        and a LeakyRelu Activation Function.

        Parameters:
            in_channels (int): The number of channels in the input
            out_channels (int): The number of channels to produce in the output
            normalize (bool): if normalization layer should be used or not

        Returns:
            torch.nn.Sequential : Single Discriminator Block including the defined layers
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1), nn.InstanceNorm2d(out_channels) if normalize else nn.Identity(), nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, img1, img2):
        """
        Forward pass of the Discriminator.

        This method applies the forward pass of the discriminator model to the input tensor x.

        Parameters:
            img1 (torch.Tensor): Input tensor of shape (batch_size, img_channels, H, W).
            img2 (torch.Tensor): Input tensor of shape (batch_size, img_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, img_channels, patch_size, patch_size).
        """
        return self.model(torch.cat([img1, img2], 1))


# Testing Code for input/ output shapes and model parameters
if __name__ == "__main__":
    from torchinfo import summary

    gen = Generator(3, 3)
    disc = Discriminator(3)

    def get_model_summary(summary_model, input_shape) -> summary:
        return summary(
            summary_model,
            input_size=input_shape,
            verbose=0,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"],
        )

    print(get_model_summary(gen, (1, 3, 256, 256)))
    print(get_model_summary(disc, ((1, 3, 256, 256), (1, 3, 256, 256))))
