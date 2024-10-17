import torch
from torch import nn
from torchinfo import summary


class Generator(nn.Module):
    def __init__(self, latent_dim: int, n_mid_channels: int, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            *self.generator_block(
                latent_dim,
                n_mid_channels * 16,
                kernel=4,
                stride=2,
                padding=1,
                normalize=False,
            ),
            *self.generator_block(n_mid_channels * 16, n_mid_channels * 8),
            *self.generator_block(n_mid_channels * 8, n_mid_channels * 4),
            *self.generator_block(n_mid_channels * 4, n_mid_channels * 2),
            *self.generator_block(n_mid_channels * 2, n_mid_channels),
            nn.ConvTranspose2d(
                n_mid_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    @staticmethod
    def generator_block(
        in_channels, out_channels, kernel=4, stride=2, padding=1, normalize=True
    ):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels) if normalize else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, n_mid_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            *self.discriminator_block(in_channels, n_mid_channels, normalize=False),
            *self.discriminator_block(n_mid_channels, n_mid_channels * 2),
            *self.discriminator_block(n_mid_channels * 2, n_mid_channels * 4),
            *self.discriminator_block(n_mid_channels * 4, n_mid_channels * 8),
            nn.Conv2d(n_mid_channels * 8, out_channels, 4, 2, 0),
            nn.Sigmoid()
        )

    @staticmethod
    def discriminator_block(in_channels, out_channels, normalize=True):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels) if normalize else nn.Identity(),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    input_x = torch.rand((128, 3, 64, 64))
    disc = Discriminator(3, 64, 1)
    print(
        summary(
            disc,
            input_size=input_x.shape,
            verbose=0,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"],
        )
    )
    print(disc(input_x).shape)
