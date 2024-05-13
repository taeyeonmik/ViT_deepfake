import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from efficientnet.model import EfficientNet


# {Norm + (MSA | MLP)} + res from residual conn
class Norm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn  # MSA or MLP

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# Position-wise Feed-Forward Networks
class FeedFoward(nn.Module):
    def __init__(self, dim:int, hidden_dim:int, dropout=0.):
        super().__init__()
        # The MLP part of the Transformer Encoder
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# Multi Head Self-Attention
class MSA(nn.Module):
    """
        dim_head = d_k (d_model / heads = 512 / 8 = 64)
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads # 512
        project_out = not (heads == 1 and dim_head == dim)  # (?)

        # Make query, key, and value by applying linear projections
        self.to_qkv_lin_proj = nn.Linear(dim, inner_dim * 3, bias=False)

        # MSA params
        self.heads = heads
        self.scale = dim_head ** -0.5  # scaling by route of d_k

        # Softmax
        self.attend = nn.Softmax(dim = -1)

        # Output after the linear projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
            b, n, h = batch, resolution, heads
        """
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv_lin_proj(x).chunk(3, dim = -1)  # chunked by 3 by the last dim
        # (h d) = h * d
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # Matmul(Q, K.T) + scaling
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # Softmax
        attn = self.attend(dots)
        # Matmul(attn, V)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out) # Linear Projection

# Vision Transformer
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        # The encoder is composed of a stack of N(depth) = 6 identical layers
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Norm(dim, MSA(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                Norm(dim, FeedFoward(dim, hidden_dim = mlp_dim, dropout = 0))
            ]))
    def forward(self, x):
        # Applying transformer's encoding for all encoder layers
        for norm_msa, norm_ff in self.layers:
            x = norm_msa(x) + x # residual conn
            x = norm_ff(x) + x  # residual conn
        return x

# Efficient Net (pretrained)
class EfficientNet_(nn.Module):
    def __init__(self, name='efficientnet-b0', vit=True):
        super().__init__()
        """ Pretrained efficientnet-b0 
            the smallest of the EfficientNet networks, 
            as a convolutional extractor for processing the input faces.
        """
        self.efficient_net = EfficientNet.from_pretrained(name)
        if not vit:
            # change the classifier
            self.efficient_net._fc = torch.nn.Linear(in_features=self.efficient_net._fc.in_features,
                                                     out_features=1,
                                                     bias=True)
        for i in range(0, len(self.efficient_net._blocks)):
            for index, param in enumerate(self.efficient_net._blocks[i].parameters()):
                if i >= len(self.efficient_net._blocks) - 3:
                    param.requires_grad = True
                    print("eff finetune True")
                else:
                    param.requires_grad = False
    def model(self):
        return self.efficient_net

# Efficient Net + ViT
class EfficientViT(nn.Module):
    def __init__(self, config, channels=512, selected_efficient_net=0, effnet=True, vit=True):
        super().__init__()
        self.vit = vit
        self.effnet = effnet

        if effnet:
            self.efficient_net = EfficientNet_(name=f'efficientnet-b{str(selected_efficient_net)}', vit=self.vit).model()
            print("Efficientnet-b" + str(selected_efficient_net) + " will be using.")
        if vit:
            image_size = config['model']['image-size'] # 224
            patch_size = config['model']['patch-size'] # 7
            num_classes = config['model']['num-classes'] # 1
            dim = config['model']['dim'] # 1024
            depth = config['model']['depth'] # 6
            heads = config['model']['heads'] # 8
            mlp_dim = config['model']['mlp-dim'] # 2048
            emb_dim = config['model']['emb-dim'] # 32
            dim_head = config['model']['dim-head'] # 64
            dropout = config['model']['dropout'] # 0.15
            emb_dropout = config['model']['emb-dropout'] # 0.15

            assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'

            num_patches = (7 // patch_size) ** 2
            """ (page 3 => https://arxiv.org/pdf/2010.11929.pdf) 
                reshape the image x (H, W, C) 
                into a sequence of flattened 2D patches x_p (N, (P P) C),
                where (H, W) is the resolution of the original image, 
                C is the number of channels, 
                (P, P) is the resolution of each image patch, 
                and N = HW/(P**2) is the resulting number of patches.
                
                The Transformer uses constant latent vector size D through all of its layers, 
                so we flatten the patches and map to D dimensions with a trainable linear projection (Eq. 1). 
                We refer to the output of this projection as the patch embeddings.
                (Eq. 1) E = (P * P * C) * D, E_pos = (N + 1) * D
            """
            patch_dim = channels * patch_size ** 2

            self.patch_size = patch_size # 7
            self.pos_embedding = nn.Parameter(torch.randn(emb_dim, 1, dim)) # ->
            self.patch_to_embedding = nn.Linear(patch_dim, dim) # -> (P * P * C) * D
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # -> (1, 1, 1024)
            self.dropout = nn.Dropout(emb_dropout) # -> 0.15
            """
                depth : 6 (encoder stacks)
                heads : 8 (nb of heads in MSA)
                dim_head : 
                mlp_dim : 2048
                dropout : 0.15
            """
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

            self.to_cls_token = nn.Identity()

            # MLP head for classification
            self.mlp_head = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, num_classes)
            )
            print("Vision Transformer will be using.")

    def forward(self, x, mask=None):
        if self.vit:
            if self.effnet:
                x = self.efficient_net.extract_features(x)
            p = self.patch_size

            x_flat2D = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
            x_proj = self.patch_to_embedding(x_flat2D)

            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x_proj), 1)
            shape = x.shape[0]
            x += self.pos_embedding[0:shape]
            x = self.dropout(x)
            x = self.transformer(x)
            x = self.to_cls_token(x[:, 0])
            x = self.mlp_head(x)
        else:
            x = self.efficient_net(x)
        return x
