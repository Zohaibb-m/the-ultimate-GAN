"""Custom Dataset for a Pix2Pix Generative Adversarial Network

This module contains custom dataset used in the Pix2Pix Generative Adversarial Network Architecture.

Classes:
    Pix2PixDataset(Dataset): Implements a Dataset for the Pix2Pix model to use as a dataloader.
"""

import glob
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image


class Pix2PixDataset(Dataset):
    """
    Dataset for the Pix2Pix GAN.

    This class implements the Pix2Pix GAN's Dataset module. It takes the root directory and loads the model for batching.
    Parameters:
        root (string): The root directory where the data resides
        transforms_ (torch.Transforms): the transformation to apply on the images
        mode (string): If the mode is train, use the whole data

    Inputs:
        index (int): The index of item to retrieve

    Returns:
        dict: The two images residing at the given index.
    """

    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms_

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        width, height = img.size
        image1 = img.crop((0, 0, int(width / 2), height))
        image2 = img.crop((int(width / 2), 0, width, height))

        if np.random.random() < 0.5:
            image1 = Image.fromarray(np.array(image1)[:, ::-1, :], "RGB")
            image2 = Image.fromarray(np.array(image2)[:, ::-1, :], "RGB")

        image1 = self.transform(image1)
        image2 = self.transform(image2)

        return {"image1": image1, "image2": image2}

    def __len__(self):
        return len(self.files)
