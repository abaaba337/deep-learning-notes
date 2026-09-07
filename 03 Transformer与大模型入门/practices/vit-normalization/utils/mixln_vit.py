import torch
from torch import nn
from math import floor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            # nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        # self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Block(nn.Module):
    def __init__(self, dim, attn, ffn, LN_type):
        super().__init__()
        assert LN_type == 'PreLN' or LN_type == 'PostLN' or LN_type == 'B2TPostLN', 'Wrong LN_type given: should be either \'PreLN\' or \'PostLN\' or \'B2TPostLN\'.'
        self.LN_type = LN_type # 'PreLN' or 'PostLN'
        self.attn = attn
        self.ffn = ffn
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x):
        if self.LN_type == 'PreLN':
            h = x
            x = self.attn_norm(x)
            x = h + self.attn(x)

            h = x
            x = self.ffn_norm(x)
            x = h + self.ffn(x)

        elif self.LN_type == 'PostLN':
            h = x
            x = self.attn(x)
            x = self.attn_norm(x+h)

            h = x
            x = self.ffn(x)
            x = self.ffn_norm(x+h)

        else:
            h1 = x
            x = self.attn(x)
            x = self.attn_norm(x+h1)

            h2 = x
            x = self.ffn(x)
            x = self.ffn_norm(x+h1+h2)
        return x


class SingleTypeLNTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., LN_type = 'PreLN'):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            attn  = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
            ffn   = FeedForward(dim, mlp_dim, dropout = dropout)
            block = Block(dim, attn, ffn, LN_type)
            self.layers.append(block)

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x

class MixTypeLNTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., alpha = .25, Mix_type = 'PostPre'):
        super().__init__()
        assert (alpha > 0.) and (alpha < 1.0) and floor(alpha*depth)>=1, 'Wrong alpha given: check if 0 < alpha < 1 and floor(alpha*depth)>=1.'
        assert Mix_type == 'PostPre' or Mix_type == 'PreB2TPost' or Mix_type == 'PrePost', 'Wrong Mix_type given: should be either \'PostPre\' or \'PreB2TPost\' or \'PrePost\' (NOT RECOMMENDED).'

        self.L_post = floor(alpha*depth)
        self.L_pre = depth - self.L_post
        self.Mix_type = Mix_type

        if Mix_type == 'PreB2TPost':
            self.post_ln_layers = SingleTypeLNTransformer(dim, self.L_post, heads, dim_head, mlp_dim, dropout, LN_type = 'B2TPostLN')
        else:
            self.post_ln_layers = SingleTypeLNTransformer(dim, self.L_post, heads, dim_head, mlp_dim, dropout, LN_type = 'PostLN')
        self.pre_ln_layers  = SingleTypeLNTransformer(dim, self.L_pre, heads, dim_head, mlp_dim, dropout, LN_type = 'PreLN')


    def forward(self, x):
        if self.Mix_type == 'PostPre':
            return self.pre_ln_layers(self.post_ln_layers(x))
        else:
            return self.post_ln_layers(self.pre_ln_layers(x))


class MixLNViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool = 'cls', channels = 3, dim_head = 64,
                 dropout = 0., emb_dropout = 0., alpha = .25, Mix_type = 'PostPre'):

        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer = MixTypeLNTransformer(dim, depth, heads, dim_head, mlp_dim, dropout, alpha, Mix_type)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
