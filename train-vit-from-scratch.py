# %% Importing libraries
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = 'mps'

# %% Loading dataset MNIST
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
    torchvision.transforms.ToTensor()
])
mnist = torchvision.datasets.MNIST(root='datasets', train=True, download=True, transform=transforms)
train_size = int(0.8 * len(mnist))
test_size = len(mnist) - train_size
train_mnist, test_mnist = torch.utils.data.random_split(mnist, [train_size, test_size])

batch_size = 64
train_loader = DataLoader(train_mnist, batch_size=64, shuffle=True)
test_loader = DataLoader(test_mnist, batch_size=64, shuffle=False)

# %% Visualize some data
for images, labels in train_loader:
    plt.figure(figsize=(8, 4), dpi=150)
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.show()
    break


# %% Data preparation for ViT
class PatchEmbedding(nn.Module):
    """
    Turn a 2D input images into a 1D sequence learnable embedding vector

    Args:
        in_channels (int): color channel
        patch_size (int): size of patches to convert input images into (16)
        embedding_dim (int): size of embedding to turn image into which is (patch_size * patch_size * in_channels)
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16,
                 embedding_dim: int = 768):
        super(PatchEmbedding, self).__init__()
