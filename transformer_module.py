import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Turn a 2D input images into a 1D sequence learnable embedding vector

    Args:
        in_channels (int): color channel
        patch_size (int): size of patches to convert input images into (16)
        embedding_dim (int): size of embedding to turn image into which is (patch_size * patch_size * in_channels)
    """

    def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim,
                                 kernel_size=patch_size, stride=patch_size, padding=0)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f'Size of input image must be divisible by patch size!'

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        # adjust to make embedding be the final dimension -> (batch_size, patch_size, embedding_dim)
        return x_flattened.permute(0, 2, 1)  # torch.Size([1, 196, 256])


# %% Working with multi-heads attention
class MultiHeadsAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int = 768, num_heads: int = 12, attention_dropout: float = 0):
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
    def __init__(self, embedding_dim: int = 768, mlp_size=3072, dropout: float = 0.1):
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


# %% Combine to create transformer block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int = 768, num_heads: int = 12, mlp_size: int = 3072,
                 mlp_dropout: float = 0.1, attention_dropout: float = 0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.msa_block = MultiHeadsAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads,
                                                  attention_dropout=attention_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x  # create residual connection for MSA block
        x = self.mlp_block(x) + x  # create residual connection for MLP block
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size: int = 224, in_channels: int = 3, patch_size: int = 16,
                 num_transformer_layers: int = 12, embedding_dim: int = 768, mlp_size: int = 3072,
                 num_heads: int = 12, attention_dropout: float = 0, mlp_dropout: float = 0.1,
                 embedding_dropout: float = 0.1, num_classes: int = 2):
        super(VisionTransformer, self).__init__()
        assert image_size // patch_size, 'Image size must be divisible by patch size'

        self.num_patches = image_size ** 2 // patch_size ** 2
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches + 1, embedding_dim))
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embedding_dim)
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderBlock(embedding_dim, num_heads, mlp_size, mlp_dropout, attention_dropout)
              for _ in range(num_transformer_layers)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x
