"""
This module defines the SimpleGAN model architecture and its different functions.

Class:
    SimpleGAN: A class that implements the SimpleGAN Model
"""

import torch
import numpy as np
import torchvision
import os

from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from the_ultimate_gan.models.simple_gan.layers import Discriminator, Generator
from tqdm import tqdm

dataset_map = {
    "mnist": datasets.MNIST,
    "fashion-mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}


class SimpleGAN:
    """Generative Adversarial Network based model to generate images from random noise.

    This class implements the Simple GAN model which is used to generate images from random noise.
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
        - image_shape (Tuple[int]): The shape of the images in the dataset.
        - orig_shape (Tuple[int]): The original shape of the images in the dataset.
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
        Initialize the Simple GAN model.

        Args:
            learning_rate (float): The learning rate for the optimizer.
            latent_dim (int): The dimension of the input noise for the generator.
            batch_size (int): The batch size to use for training.
            num_epochs (int): The number of epochs to train the model for.
            dataset (str): The name of the dataset to use for training.
        """
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_interval = checkpoint_interval
        self.resume_training = resume_training
        self.checkpoint_root_dir = "the_ultimate_gan/checkpoints/simple_gan"
        # Define the transforms
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        # Set the seed for reproducibility
        torch.manual_seed(42)
        # Generate fixed noise for generating images later
        self.fixed_noise = torch.randn((self.batch_size, self.latent_dim)).to(
            self.device
        )

        self.dataset_name = dataset  # Get the dataset name
        # Load the dataset
        self.dataset = dataset_map[dataset](
            root="Data/", transform=self.transform, download=True
        )

        self.image_shape = self.dataset.data[0].shape  # Get the shape of the images

        # Get the original shape of the images
        self.orig_shape = (
            (-1, self.image_shape[2], self.image_shape[1], self.image_shape[0])
            if len(self.image_shape) > 2
            else (-1, 1, self.image_shape[1], self.image_shape[0])
        )

        # Create a data loader
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize the generator
        self.init_generator(latent_dim, np.prod(self.image_shape))

        # Initialize the discriminator
        self.init_discriminator(np.prod(self.image_shape))

        self.init_optimizers(learning_rate)  # Initialize the optimizers

        self.init_loss_fn()  # Initialize the loss function

        self.init_summary_writers()  # Initialize the tensorboard writers

        # Print the model configurations
        print(
            f"Model Simple GAN Loaded with dataset: {dataset}. The following configurations were used:\nLearning Rate: {learning_rate}, Epochs: {num_epochs}, Batch Size: {batch_size}, Transforms with Mean: 0.5 and Std: 0.5.\n Starting the Model Training now."
        )

    def init_generator(self, latent_dim, image_dim):
        self.generator = Generator(latent_dim, image_dim).to(
            self.device
        )  # Initialize the generator

    def init_discriminator(self, image_dim):
        self.discriminator = Discriminator(image_dim).to(
            self.device
        )  # Initialize the discriminator

    def init_optimizers(self, learning_rate):
        self.opt_disc = optim.Adam(
            self.discriminator.parameters(), lr=learning_rate
        )  # Initialize the discriminator optimizer
        self.opt_gen = optim.Adam(
            self.generator.parameters(), lr=learning_rate
        )  # Initialize the generator optimizer

    def init_loss_fn(self):
        self.criterion = nn.BCELoss()  # Initialize the loss function

    def init_summary_writers(self):
        self.writer_fake = SummaryWriter(
            f"runs/GAN_{self.dataset_name}/fake"
        )  # Initialize the tensorboard writer for fake images
        self.writer_real = SummaryWriter(
            f"runs/GAN_{self.dataset_name}/real"
        )  # Initialize the tensorboard writer for real images

    def train(self):
        """
        Train the Simple GAN model using the dataset.

        This method trains the Simple GAN model using the dataset provided in the constructor.
        It trains the model for the number of epochs specified in the constructor.
        First, it trains the discriminator model and then the generator model.
        """
        try:
            step = 0  # Step for the tensorboard writer
            self.current_epoch = 1  # Initialize the current epoch
            if self.resume_training:
                self.load_model_for_training()
            while (
                self.current_epoch <= self.num_epochs
            ):  # Loop over the dataset multiple times
                for batch_idx, (real, _) in enumerate(
                    tqdm(self.loader)
                ):  # Get the inputs; data is a list of [inputs, labels]
                    real = real.view(-1, np.prod(self.image_shape)).to(
                        self.device
                    )  # Move the data to the device
                    batch_size = real.shape[0]  # Get the batch size

                    noise = torch.randn(batch_size, self.latent_dim).to(
                        self.device
                    )  # Generate random noise
                    fake = self.generator(noise)  # Generate fake images

                    discriminator_real = self.discriminator(real).view(
                        -1
                    )  # Get the discriminator output for real images
                    lossD_real = self.criterion(
                        discriminator_real, torch.ones_like(discriminator_real)
                    )  # Calculate the loss for real images

                    discriminator_fake = self.discriminator(fake).view(
                        -1
                    )  # Get the discriminator output for fake images
                    lossD_fake = self.criterion(
                        discriminator_fake, torch.zeros_like(discriminator_fake)
                    )  # Calculate the loss for fake images
                    lossD = (
                        lossD_real + lossD_fake
                    ) / 2  # Calculate the average loss for the discriminator

                    self.discriminator.zero_grad()  # Zero the gradients
                    lossD.backward(
                        retain_graph=True
                    )  # Backward pass for the discriminator
                    self.opt_disc.step()  # Update the discriminator weights

                    output = self.discriminator(fake).view(
                        -1
                    )  # Get the discriminator output for fake images
                    lossG = self.criterion(
                        output, torch.ones_like(output)
                    )  # Calculate the loss for the generator
                    self.generator.zero_grad()  # Zero the gradients
                    lossG.backward()  # Backward pass for the generator
                    self.opt_gen.step()  # Update the generator weights

                    if batch_idx == 0:
                        print(
                            f"Epoch [{self.current_epoch}/{self.num_epochs}] Loss Discriminator: {lossD:.8f}, Loss Generator: {lossG:.8f}"
                        )

                        with torch.no_grad():  # Save the generated images to tensorboard
                            fake = self.generator(self.fixed_noise).reshape(
                                self.orig_shape
                            )  # Generate fake images
                            data = real.reshape(self.orig_shape)  # Get the real images
                            img_grid_fake = torchvision.utils.make_grid(
                                fake, normalize=True
                            )  # Create a grid of fake images
                            img_grid_real = torchvision.utils.make_grid(
                                data, normalize=True
                            )  # Create a grid of real images

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
            print(
                f"An error occurred during training: {e}. Saving the model for training."
            )
        except KeyboardInterrupt:
            print("Training interrupted. Saving the model for training.")
        finally:
            self.save_model_for_training()

    def save_model_for_training(self):
        """
        Save the model for training. By training we mean that we will need some extra saved parameters other than the model's state dict to be also saved for
        resuming the training later on. We will save the model's state dict, optimizer's state dict and the current epoch number.
        """
        model_save_path = (
            f"{self.checkpoint_root_dir}/checkpoint_{self.dataset_name}_training.pt"
        )
        # Create the directory if it does not exist
        if not os.path.exists(self.checkpoint_root_dir):
            os.makedirs(self.checkpoint_root_dir)

        # Make the checkpoint dictionary
        checkpoint = {
            "epoch": self.current_epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "opt_gen_state_dict": self.opt_gen.state_dict(),
            "opt_disc_state_dict": self.opt_disc.state_dict(),
        }
        # Save the checkpoint
        torch.save(checkpoint, model_save_path)

    def load_model_for_training(self):
        """
        Load the model for training. By training we mean that we will need some extra saved parameters other than the model's state dict to be also saved for
        resuming the training later on. We will load the model's state dict, optimizer's state dict and the current epoch number.
        """

        model_save_path = (
            f"{self.checkpoint_root_dir}/checkpoint_{self.dataset_name}_training.pt"
        )
        if not os.path.exists(model_save_path):
            print(
                f"No saved model found for Simple GAN with {self.dataset_name} dataset. Skipping the loading."
            )
            return
        # Load the checkpoint
        checkpoint = torch.load(model_save_path)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.opt_gen.load_state_dict(checkpoint["opt_gen_state_dict"])
        self.opt_disc.load_state_dict(checkpoint["opt_disc_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        print("Model loaded for training.")

    def save_model_for_inference(self):
        """
        Save the model for inference. By inference we mean that we will only need the model's state dict to be saved for
        generating images later on. We will save the model's state dict.
        """
        model_save_path = (
            f"{self.checkpoint_root_dir}/checkpoint_{self.dataset_name}_inference.pt"
        )

        # Create the directory if it does not exist
        if not os.path.exists(self.checkpoint_root_dir):
            os.makedirs(self.checkpoint_root_dir)

        torch.save(self.generator.state_dict(), model_save_path)

    def load_model_for_inference(self):
        """
        Load the model for inference. By inference we mean that we will only need the model's state dict to be saved for
        generating images later on. We will load the model's state dict.
        """

        model_save_path = (
            f"{self.checkpoint_root_dir}/checkpoint_{self.dataset_name}_inference.pt"
        )
        if not os.path.exists(model_save_path):
            print(
                f"No saved model found for Simple GAN with {self.dataset_name} dataset. Skipping the loading."
            )
            return
        self.generator.load_state_dict(torch.load(model_save_path))
        print(
            f"Model loaded for inference for Simple GAN with {self.dataset_name} dataset."
        )

    def generate_images(self, num_images: int):
        """
        Generate images using the trained model.

        This method generates images using the trained generator model.
        It generates the specified number of images using the fixed noise.
        The generated images are saved to the "generated_images" folder.
        """
        import matplotlib.pyplot as plt

        if not os.path.exists("generated_images"):
            os.makedirs("generated_images")

        if not os.path.exists("generated_images/simple_gan"):
            os.makedirs("generated_images/simple_gan")

        # Load the model for inference if available
        self.load_model_for_inference()

        for i in range(num_images):
            noise = torch.randn(1, self.latent_dim).to(self.device)
            img = self.generator(noise).reshape(self.orig_shape)
            img = img.detach().cpu().numpy()
            img = img.squeeze()
            # Save the generated image
            plt.imsave(
                f"generated_images/simple_gan/generated_image_{i}.png", img, cmap="gray"
            )
