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