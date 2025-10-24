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


# -------------------------
# Transformer blocks
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(F.gelu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.dh = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        def split(t):  # [B,N,C] -> [B,h,N,dh]
            return t.view(B, N, self.h, self.dh).permute(0, 2, 1, 3)

        q = split(self.q(x)); k = split(self.k(x)); v = split(self.v(x))
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)  # [B,h,N,N]
        attn = self.attn_drop(attn.softmax(dim=-1))
        y = attn @ v                                      # [B,h,N,dh]
        y = y.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        y = self.proj_drop(self.proj(y))
        return y


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(embed_dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, int(embed_dim * mlp_ratio), drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_ratio, drop, attn_drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.tap_layers = [3, 6, 9, 12]

    def forward(self, x):
        taps = {}
        for i, blk in enumerate(self.blocks, 1):
            x = blk(x)
            if i in self.tap_layers:
                taps[i] = x
        x = self.norm(x)
        for k in self.tap_layers:
            if k not in taps: taps[k] = x
        return x, taps


# -------------------------
# Decoder (U-Net style 3D)
# -------------------------
class ConvBNReLU3D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
    def forward(self, x): return self.up(x)

class Decoder(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        # token->grid heads
        self.h12 = ConvBNReLU3D(embed_dim, 512)
        self.h09 = ConvBNReLU3D(embed_dim, 512)
        self.h06 = ConvBNReLU3D(embed_dim, 256)
        self.h03 = ConvBNReLU3D(embed_dim, 128)

        # main up path (512 → 512 → 256 → 128)
        self.up4 = Up3D(512, 512); self.fuse4 = ConvBNReLU3D(512 + 512, 512)  # meet z9
        self.up3 = Up3D(512, 512); self.fuse3 = ConvBNReLU3D(512 + 256, 512)  # meet z6
        self.up2 = Up3D(512, 512); self.fuse2 = ConvBNReLU3D(512 + 128, 256)  # meet z3
        self.up1 = Up3D(256, 256); self.fuse1 = ConvBNReLU3D(256 + 64, 128)   # meet shallow

        # skip upsamplers
        self.up09_1 = Up3D(512, 512)                         # x1
        self.up06_1 = Up3D(256, 256); self.up06_2 = Up3D(256, 256)  # x2
        self.up03_1 = Up3D(128, 128); self.up03_2 = Up3D(128, 128); self.up03_3 = Up3D(128, 128)  # x3

    @staticmethod
    def tokens_to_grid(tokens, lattice):
        B, N, E = tokens.shape
        Dp, Hp, Wp = lattice
        return tokens.transpose(1, 2).contiguous().view(B, E, Dp, Hp, Wp)

    def forward(self, taps, lattice, shallow_full):
        f12 = self.h12(self.tokens_to_grid(taps[12], lattice))  # [B,512,1,6,6]
        f09 = self.h09(self.tokens_to_grid(taps[9 ], lattice))  # [B,512,1,6,6]
        f06 = self.h06(self.tokens_to_grid(taps[6 ], lattice))  # [B,256,1,6,6]
        f03 = self.h03(self.tokens_to_grid(taps[3 ], lattice))  # [B,128,1,6,6]

        x  = self.up4(f12)                  # -> [B,512,2,12,12]
        s9 = self.up09_1(f09)               # -> [B,512,2,12,12]
        x  = self.fuse4(torch.cat([x, s9], dim=1))

        x  = self.up3(x)                    # -> [B,512,4,24,24]
        s6 = self.up06_2(self.up06_1(f06))  # -> [B,256,4,24,24]
        x  = self.fuse3(torch.cat([x, s6], dim=1))

        x  = self.up2(x)                    # -> [B,512,8,48,48]
        s3 = self.up03_3(self.up03_2(self.up03_1(f03)))  # -> [B,128,8,48,48]
        x  = self.fuse2(torch.cat([x, s3], dim=1))       # -> [B,256,8,48,48]

        x  = self.up1(x)                    # -> [B,256,16,96,96]
        x  = self.fuse1(torch.cat([x, shallow_full], dim=1))  # -> [B,128,16,96,96]
        return x


# -------------------------
# Full UNETR (paper-like sizing)
# -------------------------
class UNETR_PaperLike(nn.Module):
    def __init__(self,
                 img_size=(16, 96, 96),
                 patch_size=16,
                 in_channels=1,
                 out_channels=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 drop=0.0,
                 attn_drop=0.0):
        super().__init__()
        self.patch = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.encoder = Encoder(embed_dim, depth, num_heads, mlp_ratio, drop, attn_drop)

        # shallow full-res skip (paper uses a conv stem; keep modest here)
        self.shallow = nn.Sequential(
            ConvBNReLU3D(in_channels, 32),
            ConvBNReLU3D(32, 64),
        )

        self.decoder = Decoder(embed_dim)
        self.head = nn.Conv3d(128, out_channels, kernel_size=1)

    def forward(self, x):
        shallow_full = self.shallow(x)              # [B,64,16,96,96]
        tokens, lattice = self.patch(x)             # [B,36,768], lattice=(1,6,6)
        _, taps = self.encoder(tokens)              # taps {3,6,9,12}
        feats = self.decoder(taps, lattice, shallow_full)  # [B,128,16,96,96]
        logits = self.head(feats)                   # [B,out,16,96,96]
        return logits
