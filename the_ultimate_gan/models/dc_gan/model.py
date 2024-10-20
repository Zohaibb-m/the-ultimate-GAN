"""
This module defines the DCGAN model architecture and its different functions.

Class:
    DCGAN: A class that implements the DCGAN Model
"""

import torch
import torchvision
import os

from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from the_ultimate_gan.models.dc_gan.layers import Discriminator, Generator
from the_ultimate_gan.utils import (
    weights_init,
    save_model_for_training,
    load_model_for_training,
    save_model_for_inference,
    load_model_for_inference,
)

dataset_map = {
    "mnist": datasets.MNIST,
    "fashion-mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "celeba": datasets.CelebA,
}

class DCGAN:
    """Generative Adversarial Network based model to generate images from random noise.

    This class implements the DC GAN model which is used to generate images from random noise.
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
        Initialize the DC GAN model.

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
        self.criterion = None
        self.opt_gen = None
        self.opt_disc = None
        self.discriminator = None
        self.generator = None

        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_interval = checkpoint_interval
        self.resume_training = resume_training
        self.checkpoint_root_dir = f"the_ultimate_gan/checkpoints/{self.__class__.__name__}"
        self.out_channels = 3 if dataset not in ["mnist", "fashion-mnist"] else 1
        # Define the transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
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
        # Load the dataset
        self.dataset = dataset_map[dataset](root="Data/", transform=self.transform, download=True)
        if dataset not in dataset_map:
            print(f"Dataset: {dataset} not available for {self.__class__.__name__} Model. Try from {list(dataset_map.keys())}")
            raise Exception

        # Create a data loader
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.init_generator(latent_dim, self.out_channels)  # Initialize the generator

        self.init_discriminator(self.out_channels)  # Initialize the discriminator

        self.init_optimizers(learning_rate)  # Initialize the optimizers

        self.init_loss_fn()  # Initialize the loss function

        self.init_summary_writers()  # Initialize the tensorboard writers

        # Print the model configurations
        print(
            f"Model DC GAN Loaded with dataset: {dataset}. The following configurations were used:\nLearning Rate: {learning_rate}, Epochs: {num_epochs}, Batch Size: {batch_size}, Transforms with Mean: 0.5 and Std: 0.5 for each Channel.\n Starting the Model Training now."
        )

    def init_generator(self, latent_dim, out_channels):
        # Initialize the generator
        self.generator = Generator(latent_dim, 64, out_channels).to(self.device)
        self.generator.apply(weights_init)

    def init_discriminator(self, in_channels):
        # Initialize the discriminator
        self.discriminator = Discriminator(in_channels, 64, 1).to(self.device)
        self.discriminator.apply(weights_init)

    def init_optimizers(self, learning_rate):
        # Initialize the discriminator optimizer
        self.opt_disc = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        # Initialize the generator optimizer
        self.opt_gen = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    def init_loss_fn(self):
        self.criterion = nn.BCELoss()  # Initialize the loss function

    def init_summary_writers(self):
        # Initialize the tensorboard writer for fake images
        self.writer_fake = SummaryWriter(f"runs/DCGAN/GAN_{self.dataset_name}/fake")
        # Initialize the tensorboard writer for real images
        self.writer_real = SummaryWriter(f"runs/DCGAN/GAN_{self.dataset_name}/real")

    def train(self):
        """
        Train the DC GAN model using the dataset.

        This method trains the DC GAN model using the dataset provided in the constructor.
        It trains the model for the number of epochs specified in the constructor.
        First, it trains the discriminator model and then the generator model.
        """
        try:
            step = 0  # Step for the tensorboard writer
            self.current_epoch = 1  # Initialize the current epoch
            if self.resume_training:
                self.load_model_for_training()
            while self.current_epoch <= self.num_epochs:  # Loop over the dataset multiple times
                for batch_idx, (real, _) in enumerate(tqdm(self.loader)):  # Get the inputs; data is a list of [inputs, labels]
                    real = real.to(self.device)  # Move the data to the device
                    batch_size = real.shape[0]  # Get the batch size

                    # Generate random noise
                    noise = torch.randn((batch_size, self.latent_dim, 1, 1)).to(self.device)
                    fake = self.generator(noise)  # Generate fake images

                    discriminator_real = self.discriminator(real)
                    # Calculate the loss for real images
                    loss_d_real = self.criterion(discriminator_real, torch.ones_like(discriminator_real))

                    discriminator_fake = self.discriminator(fake).view(-1)  # Get the discriminator output for fake images

                    # Calculate the loss for fake images
                    loss_d_fake = self.criterion(discriminator_fake, torch.zeros_like(discriminator_fake))

                    # Calculate the average loss for the discriminator
                    loss_discriminator = (loss_d_real + loss_d_fake) / 2

                    self.discriminator.zero_grad()  # Zero the gradients
                    loss_discriminator.backward(retain_graph=True)  # Backward pass for the discriminator
                    self.opt_disc.step()  # Update the discriminator weights

                    output = self.discriminator(fake).view(-1)  # Get the discriminator output for fake images
                    loss_generator = self.criterion(output, torch.ones_like(output))  # Calculate the loss for the generator
                    self.generator.zero_grad()  # Zero the gradients
                    loss_generator.backward()  # Backward pass for the generator
                    self.opt_gen.step()  # Update the generator weights

                    if batch_idx == 0:
                        print(f"Epoch [{self.current_epoch}/{self.num_epochs}] Loss Discriminator: {loss_discriminator:.8f}, Loss Generator: {loss_generator:.8f}")

                        with torch.no_grad():  # Save the generated images to tensorboard
                            fake = self.generator(self.fixed_noise)  # Generate fake images
                            data = real  # Get the real images
                            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)  # Create a grid of fake images
                            img_grid_real = torchvision.utils.make_grid(data, normalize=True)  # Create a grid of real images

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
