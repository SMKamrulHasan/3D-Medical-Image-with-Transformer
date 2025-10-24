
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
