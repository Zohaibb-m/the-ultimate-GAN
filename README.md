<div align="center">

# The Ultimate GAN

<img width="61.8%" alt="TUGAN: The ultimate GAN" src="https://media.licdn.com/dms/image/D4D12AQFcqXuUKAq68g/article-cover_image-shrink_600_2000/0/1686287661420?e=2147483647&v=beta&t=rgvuNTarPWthtcPydVAfkFpI4Jkr9L6Ed_tkF2PnNiE">

<br>
</br>

</div>

---

1. [Overview](#overview)
2. [**Installation** `🛠️`](#how-to-install-the-package)
3. [**Usage**](#usage)

## Overview

This library focuses on different variants of the Generative Adversarial Networks. This library was created as a learning initiative to learn everything there is about GANs. The goal of this library is to provide all types of GANs under one hood and let anyone explore, train and learn about them with just one command. I will also try to write a blog on medium regarding each one of the GANs mentioned in this library.

## How to install the package
I am yet to add the pyproject toml file for direct or indirect pip installations. So for now you can clone the repository and then install the requirements using
```bash
git clone https://github.com/Zohaibb-m/the-ultimate-GAN.git
cd the-ultimate-GAN
pip install -r requirements.txt
```

## Usage
I will soon add the detailed usage docs but for now we can run and train the gans using:

```bash
python3 -m the_ultimate_gan.cli train --model-name s-gan --dataset fashion-mnist --learning-rate 0.0003
```

To start a tensorboard session run:
```bash
tensorboard --logdir ./runs
```

More options are available and can be viewed using
```bash
python3 -m the_ultimate_gan.cli train --help
```
<hr>

## Implementations

### Simple GAN
#### Explanation:
The medium article explaining Simple GAN is available with the same code explanation at <kbd>[Building a Simple GAN](https://medium.com/@zohaibmunir32/generative-adversarial-networks-part-1-an-introduction-and-implementation-of-the-simplest-form-6b8973df38fd)</kbd>

#### Training
Trained on Apple M2 16gb (Average 89it/s) with the following configurations:</br>
Learning Rate: 0.0003, Epochs: 50, Batch Size: 32, Transforms with Mean: 0.5 and Std: 0.5
```bash
python3 -m the_ultimate_gan.cli train --model-name s-gan --dataset fashion-mnist -rt
```
<p> <img src="assets/sgan-mnist.gif" width="360" > &nbsp &nbsp &nbsp <img src="assets/sgan-fashion-mnist.gif" width="360"> </p>

<hr>

### Deep Convolutional GAN
#### Explanation:
The medium article explaining DC GAN is available with the same code explanation at <kbd>[Building a Deep Convolutional GAN](https://medium.com/@zohaibmunir32/generative-adversarial-networks-part-2-improved-image-generation-using-deep-convolutional-gan-6e472623831e)</kbd>

#### Training
Trained on Nvidia GPU P100 (Average 6it/s) with the following configurations:</br>
Learning Rate: 0.0002, Epochs: 50, Batch Size: 128, Transforms with Mean: 0.5 and Std: 0.5 for each Channel.
```bash
python3 -m the_ultimate_gan.cli train --model-name dc-gan --dataset mnist
```
<p> <img src="assets/dcgan-mnist.gif" width="360" ></p>

<hr>

## Upcoming/Roadmap

<details open>
<summary>WGAN</summary>

```python
# Next I will try to implement WGan this weekend along with the blog
```

</details>