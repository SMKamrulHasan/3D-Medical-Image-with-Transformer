
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
