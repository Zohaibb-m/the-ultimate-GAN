"""
This module defines the WGAN-GP model architecture and its different functions.

Class:
    WGAN: A class that implements the WGAN-GP Model
"""

import torch
import torchvision

from tqdm import tqdm
from the_ultimate_gan.models.w_gan.model import WGAN
from the_ultimate_gan.utils import gradient_penalty


class WGANGP(WGAN):
    """Generative Adversarial Network based model to generate images from random noise.

    This class extends the WGAN with a gradient penalty to correctly implement the lipschitz clipping technique.
    It uses a Generator and Critic model to train the GAN and generate images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_gradient_penalty = 10

    def init_optimizers(self, learning_rate):
        self.opt_gen = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))

    def train(self):
        """
        Train the WGAN GP model using the dataset.

        This method trains the WGAN GP model using the dataset provided in the constructor.
        It trains the model for the number of epochs specified in the constructor.
        First, it trains the critic model and then the generator model.
        """
        try:
            step = 0  # Step for the tensorboard writer
            self.current_epoch = 1  # Initialize the current epoch
            if self.resume_training:
                self.load_model_for_training()

            self.critic.train()
            self.generator.train()

            # Loop over the dataset multiple times
            while self.current_epoch <= self.num_epochs:
                for batch_idx, (real, _) in enumerate(tqdm(self.loader)):  # Get the inputs; data is a list of [inputs, labels]
                    real = real.to(self.device)  # Move the data to the device
                    batch_size = real.shape[0]  # Get the batch size
                    loss_critic = 0

                    for i in range(self.critic_steps):
                        # Generate random noise
                        noise = torch.randn((batch_size, self.latent_dim, 1, 1)).to(self.device)
                        fake = self.generator(noise)  # Generate fake images

                        critic_real = self.critic(real)
                        critic_fake = self.critic(fake)
                        # Calculate the gradient penalty
                        gp = gradient_penalty(self.critic, real, fake, device=self.device)
                        loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.lambda_gradient_penalty * gp
                        self.critic.zero_grad()  # Zero the gradients
                        # Backward pass for the critic
                        loss_critic.backward(retain_graph=True)
                        self.opt_critic.step()  # Update the critic weights

                    output = self.critic(fake)  # Get the critic output for fake images
                    loss_generator = -torch.mean(output)
                    self.generator.zero_grad()  # Zero the gradients
                    loss_generator.backward()  # Backward pass for the generator
                    self.opt_gen.step()  # Update the generator weights

                    if batch_idx % 100 == 0 and batch_idx > 0:
                        print(f"Epoch [{self.current_epoch}/{self.num_epochs}] Loss Critic: {loss_critic:.8f}, Loss Generator: {loss_generator:.8f}")

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
