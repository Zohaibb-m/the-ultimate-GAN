"""
This module defines the LSGAN model architecture and its different functions.

Class:
    LSGAN: A class that implements the LSGAN Model
"""

from torch import nn

from the_ultimate_gan.models.dc_gan.model import DCGAN
from the_ultimate_gan.models.dc_gan.layers import Discriminator
from the_ultimate_gan.utils import weights_init


class LSGAN(DCGAN):
    """Generative Adversarial Network based model to generate images from random noise.

    This class implements the LS GAN model which is used to generate images from random noise.
    It uses a Generator and Discriminator model to train the GAN and generate images.
    This class inherits the DCGAN but just to solve its problem of vanishing gradient, we use
    MSE Loss instead of BCE Loss. Everything else is same as DCGAN.
    """

    def init_loss_fn(self):
        self.criterion = nn.MSELoss()  # Initialize the loss function

    def init_discriminator(self, in_channels):
        # Initialize the discriminator
        self.discriminator = Discriminator(in_channels, 64, 1, last_activation=False).to(self.device)
        self.discriminator.apply(weights_init)
