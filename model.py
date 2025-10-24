
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
