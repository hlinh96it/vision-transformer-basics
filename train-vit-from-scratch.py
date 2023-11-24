# %% Importing libraries
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt

from patch_embedding import PatchEmbedding

device = 'mps'
    
    
# %% Working with multi-heads attention
class MultiHeadsAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int=768, num_heads: int=12, attention_dropout: float=0):
        super(MultiHeadsAttentionBlock, self).__init__()
        self.layer_normalization = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multi_heads_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads,
                                                           dropout=attention_dropout, batch_first=True)
        
    def forward(self, x):
        x = self.layer_normalization(x)
        attention_output, _ = self.multi_heads_attention(query=x, key=x, value=x, need_weights=False)
        return attention_output
        
# %% Working with MLP layer
class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int=768, mlp_size=3072, dropout: float=0.1):
        super(MLPBlock, self).__init__()
        self.layer_normalization = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )
        
    def forward(self, x):
        x = self.layer_normalization(x)
        x = self.mlp(x)
        return x
    
# %% Combine together to create transformer block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int=768, num_heads: int=12, mlp_size: int=3072, 
                 mlp_dropout: float=0.1, attention_dropout: float=0.1):
        super(TransformerEncoderBlock, self)
        self.msa_block = MultiHeadsAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads,
                                                  attention_dropout=attention_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)
        
    def forward(self, x):
        x = self.msa_block(x) + x  # create residual connection for MSA block
        x = self.msa_block(x) + x  # create residual connection for MLP block
        return x


# %% Hyper-parameter configurations
IMG_SIZE = 224
train_test_split = 0.8
batch_size = 64
patch_size = 16
embedding_dim = 16 * 16

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


# %% Data preparation for ViT
patchify = PatchEmbedding(in_channels=1, patch_size=patch_size, embedding_dim=embedding_dim)
class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dim), requires_grad=True)


# %% Transformer model
transformer_encoder_block = TransformerEncoderBlock(embedding_dim=embedding_dim)
summary(transformer_encoder_block, input_size=(1, 197, embedding_dim),
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=20, row_settings=['var_names'])