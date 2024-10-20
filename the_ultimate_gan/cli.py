import click

from the_ultimate_gan.models.simple_gan.model import SimpleGAN
from the_ultimate_gan.models.dc_gan.model import DCGAN
from the_ultimate_gan.models.w_gan.model import WGAN
from the_ultimate_gan.models.w_gan_gp.model import WGANGP

model_map = {"s-gan": SimpleGAN, "dc-gan": DCGAN, "w-gan": WGAN, "w-gan-gp": WGANGP}

@click.group()
def tugan():
    """
    The Ultimate GAN\n
    Learn, Train and explore Generative Adversarial Networks.
    """

@tugan.command(no_args_is_help=True)
@click.option(
    "--model-name",
    "-m",
    required=True,
    help="The GAN Model you want to train",
    type=click.Choice(["s-gan", "dc-gan", "w-gan", "w-gan-gp"]),
)
@click.option(
    "--dataset",
    "-d",
    default="mnist",
    help="The dataset you want to use for training.",
    type=click.Choice(["mnist", "cifar10", "fashion-mnist", "cifar100", "celeba"]),
)
@click.option(
    "--learning-rate",
    "-lr",
    default=3e-4,
    help="The Learning Rate for the optimizer",
    type=float,
)
@click.option(
    "--latent-dim",
    "-ld",
    default=64,
    help="The Latent Dimension for the random noise",
    type=int,
)
@click.option("--batch-size", "-bs", default=32, help="Batch Size to use", type=int)
@click.option(
    "--num-epochs",
    "-ne",
    default=100,
    help="Number of epochs to train model for",
    type=int,
)
@click.option(
    "--checkpoint-interval",
    "-ci",
    default=10,
    help="Interval to save model checkpoints",
    type=int,
)
@click.option(
    "--resume-training",
    "-rt",
    default=False,
    help="Resume training from a checkpoint",
    is_flag=True,
)
def train(
    model_name,
    dataset,
    learning_rate,
    latent_dim,
    batch_size,
    num_epochs,
    checkpoint_interval,
    resume_training,
):
    try:
        model = model_map[model_name](
            learning_rate,
            latent_dim,
            batch_size,
            num_epochs,
            dataset,
            checkpoint_interval,
            resume_training,
        )
        model.train()
    except Exception as e:
        print(f"Model Training interrupted due to an error.")

if __name__ == "__main__":
    tugan()
