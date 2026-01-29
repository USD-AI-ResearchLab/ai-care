import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    def __init__(self, n_patches, hidden_dim, token_dim, channel_dim):
        super().__init__()

        # Token-mixing MLP (operates on patches)
        self.token_norm = nn.LayerNorm(n_patches)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_patches, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, n_patches),
        )

        # Channel-mixing MLP (operates on channels)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, hidden_dim),
        )

    def forward(self, x):
        # x: (B, N, D)

        # Token mixing
        y = self.token_norm(x.transpose(1, 2))   # (B, D, N)
        y = self.token_mlp(y)
        x = x + y.transpose(1, 2)

        # Channel mixing
        y = self.channel_norm(x)
        y = self.channel_mlp(y)
        x = x + y

        return x


class MLPMixer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        num_classes,
        hidden_dim=128,
        depth=4,
        token_dim=64,
        channel_dim=256,
    ):
        super().__init__()

        assert image_size % patch_size == 0
        n_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.blocks = nn.Sequential(*[
            MixerBlock(
                n_patches=n_patches,
                hidden_dim=hidden_dim,
                token_dim=token_dim,
                channel_dim=channel_dim,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x)                  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)         # (B, N, D)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)                        # global average pooling
        return self.head(x)
