"""
This module defines the Pix2Pix GAN model architecture and its different functions.

Class:
    Pix2Pix GAN: A class that implements the Pix2Pix GAN Model
"""

from configparser import Interpolation

import torch
import torchvision
import os

from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from the_ultimate_gan.models.pix2pix.layers import Discriminator, Generator
from the_ultimate_gan.models.pix2pix.dataset import Pix2PixDataset
from the_ultimate_gan.utils import (
    weights_init,
    save_model_for_training,
    load_model_for_training,
    save_model_for_inference,
    load_model_for_inference,
)


class Pix2PixModel:
    """Generative Adversarial Network based model to generate a certain styled image from another image.

    This class implements the PIX2PIX GAN  model which is used to generate images.
    It uses a Generator and Discriminator model to train the GAN and generate images.

    Attributes:
        - device (torch.device): The device to run the model on.
        - latent_dim (int): The dimension of the input noise for the generator.
        - batch_size (int): The batch size to use for training.
        - num_epochs (int): The number of epochs to train the model for.
        - transform (torchvision.transforms.Compose): The transforms to apply to the input data.
        - fixed_noise (torch.Tensor): The fixed noise to use for generating images.
        - dataset_name (str): The name of the dataset to use for training.
        - dataset (torchvision.datasets): The dataset to use for training.
        - loader (torch.utils.data.DataLoader): The data loader to use for loading data.
        - generator (Generator): The generator model to use for generating images.
        - discriminator (Discriminator): The discriminator model to use for detecting fake or real images.
        - opt_disc (torch.optim.Adam): The optimizer to use for the discriminator model.
        - opt_gen (torch.optim.Adam): The optimizer to use for the generator model.
        - criterion (torch.nn.BCELoss): The loss function to use for training.
        - writer_fake (torch.utils.tensorboard.SummaryWriter): The tensorboard writer for fake images.
        - writer_real (torch.utils.tensorboard.SummaryWriter): The tensorboard writer for real images.

    Methods:
        - init_generator(latent_dim, image_dim): Initialize the generator model.
        - init_discriminator(image_dim): Initialize the discriminator model.
        - init_optimizers(learning_rate): Initialize the optimizers for the models.
        - init_loss_fn(): Initialize the loss function for training.
        - init_summary_writers(): Initialize the tensorboard summary writers.
        - train(): Train the model using the dataset.
    """

    def __init__(
        self,
        learning_rate: float,
        latent_dim: int,
        batch_size: int,
        num_epochs: int,
        dataset: str,
        checkpoint_interval: int,
        resume_training: bool,
    ):
        """
        Initialize the Pix2Pix  GAN model.

        Args:
            learning_rate (float): The learning rate for the optimizer.
            latent_dim (int): The dimension of the input noise for the generator.
            batch_size (int): The batch size to use for training.
            num_epochs (int): The number of epochs to train the model for.
            dataset (str): The name of the dataset to use for training.
        """
        self.current_epoch = None
        self.writer_real = None
        self.writer_fake = None
        self.criterion_gan = None
        self.criterion_pixelwise = None
        self.opt_gen = None
        self.opt_disc = None
        self.discriminator = None
        self.generator = None

        # Loss weight of L1 pixel-wise loss between translated image and real image
        self.lambda_pixel = 100

        # Calculate the Discriminator (Patch GAN) output
        self.patch = (1, 256 // 2**4, 256 // 2**4)

        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_interval = checkpoint_interval
        self.resume_training = resume_training
        self.checkpoint_root_dir = f"the_ultimate_gan/checkpoints/{self.__class__.__name__}"
        self.in_channels = self.out_channels = 3
        # Define the transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(self.out_channels)],
                    [0.5 for _ in range(self.out_channels)],
                ),
            ]
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # Set the seed for reproducibility
        torch.manual_seed(42)
        # Generate fixed noise for generating images later
        self.fixed_noise = torch.randn((self.batch_size, self.latent_dim, 1, 1)).to(self.device)

        self.dataset_name = dataset  # Get the dataset name

        # Create a data loader
        self.loader = DataLoader(Pix2PixDataset("Data/maps", transforms_=self.transform), batch_size=self.batch_size, shuffle=True)

        self.init_generator(self.in_channels, self.out_channels)  # Initialize the generator

        self.init_discriminator(self.in_channels)  # Initialize the discriminator

        self.init_optimizers(learning_rate)  # Initialize the optimizers

        self.init_loss_fn()  # Initialize the loss function

        self.init_summary_writers()  # Initialize the tensorboard writers

        # Print the model configurations
        print(
            f"Model Pix2Pix Loaded with dataset: Maps. The following configurations were used:\nLearning Rate: {learning_rate}, Epochs: {num_epochs}, Batch Size: {batch_size}, Transforms with Mean: 0.5 and Std: 0.5 for each Channel.\n Starting the Model Training now."
        )

    def init_generator(self, in_channels, out_channels):
        # Initialize the generator
        self.generator = Generator(in_channels, out_channels).to(self.device)
        self.generator.apply(weights_init)

    def init_discriminator(self, in_channels):
        # Initialize the discriminator
        self.discriminator = Discriminator(in_channels).to(self.device)
        self.discriminator.apply(weights_init)

    def init_optimizers(self, learning_rate):
        # Initialize the discriminator optimizer
        self.opt_disc = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        # Initialize the generator optimizer
        self.opt_gen = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    def init_loss_fn(self):
        self.criterion_gan = nn.MSELoss()  # Initialize the GAN loss function
        self.criterion_pixelwise = nn.L1Loss()  # Initialize the Pixelwise loss function

    def init_summary_writers(self):
        # Initialize the tensorboard writer for fake images
        self.writer_fake = SummaryWriter(f"runs/{self.__class__.__name__}/{self.dataset_name}/fake")
        # Initialize the tensorboard writer for real images
        self.writer_real = SummaryWriter(f"runs/{self.__class__.__name__}/{self.dataset_name}/real")

    def train(self):
        """
        Train the Pix2Pix GAN model using the dataset.

        This method trains the Pix2Pix GAN model using the dataset provided in the constructor.
        It trains the model for the number of epochs specified in the constructor.
        First, it trains the generator model and then the discriminator model.
        """
        try:
            step = 0  # Step for the tensorboard writer
            self.current_epoch = 1  # Initialize the current epoch
            if self.resume_training:
                self.load_model_for_training()
            while self.current_epoch <= self.num_epochs:  # Loop over the dataset multiple times
                for batch_idx, real in enumerate(tqdm(self.loader)):  # Get the inputs; data is a list of [inputs, labels]
                    real_1 = real["image1"].to(self.device)  # Move the data to the device
                    real_2 = real["image2"].to(self.device)  # Move the second image to the device

                    self.opt_gen.zero_grad()  # Zero the gradients

                    fake_image2 = self.generator(real_1)  # Generate fake images

                    disc_fake = self.discriminator(fake_image2, real_1)

                    loss_gan = self.criterion_gan(disc_fake, torch.ones((real_2.shape[0], *self.patch), device=self.device, requires_grad=False))

                    # Pixel Wise Loss
                    loss_pixel = self.criterion_pixelwise(fake_image2, real_2)

                    # Total Loss
                    loss_gen = loss_gan + self.lambda_pixel * loss_pixel

                    loss_gen.backward()

                    self.opt_gen.step()

                    # Train Discriminator

                    self.opt_disc.zero_grad()

                    disc_real = self.discriminator(real_2, real_1)
                    loss_real = self.criterion_gan(disc_real, torch.ones((real_1.shape[0], *self.patch), device=self.device, requires_grad=False))

                    disc_fake = self.discriminator(fake_image2.detach(), real_1)
                    loss_fake = self.criterion_gan(disc_fake, torch.zeros((real_1.shape[0], *self.patch), device=self.device, requires_grad=False))

                    loss_disc = 0.5 * (loss_real + loss_fake)

                    loss_disc.backward()

                    self.opt_disc.step()

                    if batch_idx % (self.batch_size // 10) == 0 and batch_idx > 0:
                        print(f"Epoch [{self.current_epoch}/{self.num_epochs}] Loss Discriminator: {loss_disc:.8f}, Loss Generator: {loss_gen:.8f}")

                        with torch.no_grad():  # Save the generated images to tensorboard
                            fake = self.generator(real_1)  # Generate fake images
                            data = real  # Get the real images
                            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)  # Create a grid of fake images
                            img_grid_real = torchvision.utils.make_grid(real_2, normalize=True)  # Create a grid of real images

                            self.writer_fake.add_image(
                                f"{self.dataset_name} Fake Images",
                                img_grid_fake,
                                global_step=step,
                            )  # Add the fake images to tensorboard
                            self.writer_real.add_image(
                                f"{self.dataset_name} Real Images",
                                img_grid_real,
                                global_step=step,
                            )  # Add the real images to tensorboard
                            step += 1  # Increment the step
                if self.current_epoch % self.checkpoint_interval == 0:
                    self.save_model_for_inference()
                    print(f"Model saved at epoch {self.current_epoch} for inference.")
                self.current_epoch += 1  # Increment the epoch
        except Exception as e:
            print(f"An error occurred during training: {e}. Saving the model for training.")
        except KeyboardInterrupt:
            print("Training interrupted. Saving the model for training.")
        finally:
            self.save_model_for_training()

    def save_model_for_training(self):
        save_model_for_training(
            self.checkpoint_root_dir,
            self.dataset_name,
            self.current_epoch,
            self.generator.state_dict(),
            self.discriminator.state_dict(),
            self.opt_gen.state_dict(),
            self.opt_disc.state_dict(),
        )

    def load_model_for_training(self):
        # Load the checkpoint
        checkpoint = load_model_for_training(self.checkpoint_root_dir, self.dataset_name, self.__class__.__name__, self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.opt_gen.load_state_dict(checkpoint["opt_gen_state_dict"])
        self.opt_disc.load_state_dict(checkpoint["opt_disc_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        print("Model loaded for training.")

    def save_model_for_inference(self):
        save_model_for_inference(self.checkpoint_root_dir, self.dataset_name, self.generator.state_dict())

    def load_model_for_inference(self):
        generator_state_dict = load_model_for_inference(self.checkpoint_root_dir, self.dataset_name, self.__class__.__name__, self.device)
        self.generator.load_state_dict(generator_state_dict)
        print(f"Model loaded for inference for Simple GAN with {self.dataset_name} dataset.")
