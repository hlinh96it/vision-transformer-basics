# %% Importing libraries
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt

from transformer_module import *

device = 'mps'


# %% Hyper-parameter configurations
IMG_SIZE, num_patches = 224, 16
train_test_split = 0.8
batch_size = 64
patch_size = IMG_SIZE // num_patches
embedding_dim = num_patches * num_patches


# %% Loading dataset MNIST
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
    torchvision.transforms.ToTensor()
])
mnist = torchvision.datasets.MNIST(root='datasets', train=True, download=True, transform=transforms)
train_size = int(train_test_split * len(mnist))
test_size = len(mnist) - train_size
train_mnist, test_mnist = torch.utils.data.random_split(mnist, [train_size, test_size])

train_loader = DataLoader(train_mnist, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_mnist, batch_size=batch_size, shuffle=False)


# %% Transformer model
transformer_encoder_block = TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=4)
summary(transformer_encoder_block, input_size=(1, patch_size, embedding_dim),
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=20, row_settings=['var_names'])

transformer_model = VisionTransformer(IMG_SIZE, in_channels=1, patch_size=patch_size, num_transformer_layers=12,
                                      embedding_dim=embedding_dim, mlp_size=3072, num_heads=4,
                                      attention_dropout=0.1, mlp_dropout=0.1, embedding_dropout=0.1,
                                      num_classes=10)
