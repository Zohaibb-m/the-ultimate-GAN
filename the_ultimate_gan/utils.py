import os
import torch

from torch import nn
from torchinfo import summary

def weights_init(m):
    """
    This function uses the technique discussed in the DC Gan's paper to initialize weights with a specific mean and std.
    Args:
        m: The model to initialize the weights for

    Returns:
        None
    """
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_model_summary(summary_model, input_shape) -> summary:
    """
    Find the model summary (parameters, input shape, output shape, trainable parameters etc.)
    Args:
        summary_model (nn.Module): The model for which we want to print the summary details
        input_shape (shape): The input shape that the model takes

    Returns:
        A summary object with the model details in it
    """
    return summary(
        summary_model,
        input_size=input_shape,
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )

def save_model_for_training(
    checkpoint_root_dir,
    dataset_name,
    current_epoch,
    generator_state_dict,
    discriminator_state_dict,
    opt_gen_state_dict,
    opt_disc_state_dict,
):
    """
    Save the model for training. By training, we mean that we will need some extra saved parameters other than the model's state dict to be also saved for
    resuming the training later on. We will save the model's state dict, optimizer's state dict and the current epoch number.
    """
    model_save_path = f"{checkpoint_root_dir}/checkpoint_{dataset_name}_training.pt"
    # Create the directory if it does not exist
    if not os.path.exists(checkpoint_root_dir):
        os.makedirs(checkpoint_root_dir)

    # Make the checkpoint dictionary
    checkpoint = {
        "epoch": current_epoch,
        "generator_state_dict": generator_state_dict,
        "discriminator_state_dict": discriminator_state_dict,
        "opt_gen_state_dict": opt_gen_state_dict,
        "opt_disc_state_dict": opt_disc_state_dict,
    }
    # Save the checkpoint
    torch.save(checkpoint, model_save_path)

def load_model_for_training(checkpoint_root_dir, dataset_name, class_name, device):
    """
    Load the model for training. By training, we mean that we will need some extra saved parameters other than the model's state dict to be also saved for
    resuming the training later on. We will load the model's state dict, optimizer's state dict and the current epoch number.
    """

    model_save_path = f"{checkpoint_root_dir}/checkpoint_{dataset_name}_training.pt"
    if not os.path.exists(model_save_path):
        print(f"No saved model found for {class_name} with {dataset_name} dataset. Skipping the loading.")
        return
    # Load the checkpoint and return it
    return torch.load(model_save_path, map_location=device)

def save_model_for_inference(checkpoint_root_dir, dataset_name, generator_state_dict):
    """
    Save the model for inference. By inference, we mean that we will only need the model's state dict to be saved for
    generating images later on. We will save the model's state dict.
    """
    model_save_path = f"{checkpoint_root_dir}/checkpoint_{dataset_name}_inference.pt"

    # Create the directory if it does not exist
    if not os.path.exists(checkpoint_root_dir):
        os.makedirs(checkpoint_root_dir)

    torch.save(generator_state_dict, model_save_path)

def load_model_for_inference(checkpoint_root_dir, dataset_name, class_name, device):
    """
    Load the model for inference. By inference, we mean that we will only need the model's state dict to be saved for
    generating images later on. We will load the model's state dict.
    """

    model_save_path = f"{checkpoint_root_dir}/checkpoint_{dataset_name}_inference.pt"
    if not os.path.exists(model_save_path):
        print(f"No saved model found for {class_name} with {dataset_name} dataset. Skipping the loading.")
        return
    model = torch.load(model_save_path, map_location=device)
    print(f"Model loaded for inference for {class_name} with {dataset_name} dataset.")
    return model

def generate_images(num_images, generator, class_name, latent_dim, device, orig_shape):
    """
    Generate images using the trained model.

    This method generates images using the trained generator model.
    It generates the specified number of images using the fixed noise.
    The generated images are saved to the "generated_images" folder.
    """
    import matplotlib.pyplot as plt

    if not os.path.exists("generated_images"):
        os.makedirs("generated_images")

    if not os.path.exists(f"generated_images/{class_name}"):
        os.makedirs(f"generated_images/{class_name}")

    for i in range(num_images):
        noise = torch.randn(1, latent_dim).to(device)
        img = generator(noise).reshape(orig_shape)
        img = img.detach().cpu().numpy()
        img = img.squeeze()
        # Save the generated image
        plt.imsave(
            f"generated_images/{class_name}/generated_image_{i}.png",
            img,
        )

def gradient_penalty(critic, real, fake, device="cpu"):
    batch_size, channels, height, width = real.shape
    alpha = torch.rand((batch_size, 1, 1, 1)).repeat(1, channels, height, width).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    grad_pen = torch.mean((gradient_norm - 1) ** 2)
    return grad_pen
