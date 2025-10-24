import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Patch Embedding (3D) + learnable pos_embed
# -------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(16, 96, 96), patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.D, self.H, self.W = img_size
        self.P = patch_size
        assert self.D % self.P == 0 and self.H % self.P == 0 and self.W % self.P == 0
        self.Dp, self.Hp, self.Wp = self.D // self.P, self.H // self.P, self.W // self.P
        self.num_patches = self.Dp * self.Hp * self.Wp

        self.proj = nn.Linear((self.P ** 3) * in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        B, C, D, H, W = x.shape
        P = self.P
        x = x.view(B, C, self.Dp, P, self.Hp, P, self.Wp, P)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.view(B, self.Dp, self.Hp, self.Wp, -1)
        x = x.view(B, self.num_patches, -1)
        tokens = self.proj(x) + self.pos_embed  # [B, N, 768]
        return tokens, (self.Dp, self.Hp, self.Wp)
