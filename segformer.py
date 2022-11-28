"""Segformer model
"""


# DL library imports
import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

"""
Segformer Network
Encoder:
 - Overlap Patch Embedding
 - Efficient Self-Attention
 - Mix FFNs
Decoder Head
"""

class overlap_patch_embed(nn.Module):
    def __init__(self, patch_size, stride, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size // 2, patch_size // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, h, w

class efficient_self_attention(nn.Module):
    def __init__(self, attn_dim, num_heads, dropout_p, sr_ratio):
        super().__init__()
        assert attn_dim % num_heads == 0, f'expected attn_dim {attn_dim} to be a multiple of num_heads {num_heads}'
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(attn_dim, attn_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(attn_dim)

        # Multi-head Self-Attention using dot product
        # Query - Key Dot product is scaled by root of head_dim
        self.q = nn.Linear(attn_dim, attn_dim, bias=True)
        self.kv = nn.Linear(attn_dim, attn_dim * 2, bias=True)
        self.scale = (attn_dim // num_heads) ** -0.5

        # Projecting concatenated outputs from 
        # multiple heads to single `attn_dim` size
        self.proj = nn.Linear(attn_dim, attn_dim)
        
        # dropout
        self.proj_drop = nn.Dropout(self.dropout_p)

    def forward(self, x, h, w):
        B, N, C = x.shape
        # [B, N, C] -> [B, N, num_heads, embed_dim] -> [B, num_heads, N, embed_dim]
        q = self.q(x)
        q = rearrange(q, 'b n (h e) -> b n h e', h=self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # since self.sr is doing Conv2d, we need shape = [B, C, H, W] to do that
            x = rearrange(x, 'b (h w) c -> b c h w', h=h)
            x = self.sr(x)
            # reshape from [B, C, H, W] back to [B, N, C]
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)

        # [B, N, C*2] -> [B, N, 2, num_heads, embed_dim] -> [2, B, num_heads, N, embed_dim]
        kv = self.kv(x)
        kv = rearrange(kv, 'b n (s h e) -> b n s h e', s=2, h=self.num_heads).permute(2, 0, 3, 1, 4)
        
        # seperate kv. k = [B, num_heads, N, embed_dim], v = [B, num_heads, N, embed_dim]
        k, v = kv.unbind(0)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)

        output = attn @ v
        # reshape from [B, num_heads, N, embed_dim] to [B, N, C] (C = concatenate embed_dim along num_heads)
        # output_ = output.transpose(1,2).reshape(B,N,C)
        output = rearrange(output, 'b h n e -> b n (h e)')
        output = self.proj(output)  # (attn_dim, attn_dim)
        output = self.proj_drop(output)

        # save the q,k,v,attn for visualization
        # attn_output = {'key' : k, 'query' : q, 'value' : v, 'attn' : attn}
        attn_output = {'attn' : attn}
        return output, attn_output

class mix_feedforward(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, dropout_p = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        # using depthwise convolution
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, 
                                  padding=1, groups=hidden_features)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x, h, w):
        # MLP 1
        output = self.fc1(x)  # [B, N, C]
        # reshape from [B, N, C] to [B, C, H, W] for Conv2d
        output = rearrange(output, 'b (h w) c -> b c h w', h=h)
        output = self.conv(output)
        # reshape back to [B, N, C]
        output = rearrange(output, 'b c h w -> b (h w) c')
        # GELU
        output = self.gelu(output)
        output = self.drop(output)
        # MLP 2
        output = self.fc2(output)
        output = self.drop(output)
        return output

class transformer_block(nn.Module):
    def __init__(self, dim, num_heads, dropout_p, drop_path_p, sr_ratio):
        super().__init__()
        # One transformer block is defined as :
        # Norm -> efficient self-attention -> Norm -> Mix-FFN
        # skip-connections are added after attention and FF layers
        self.attn = efficient_self_attention(attn_dim=dim, num_heads=num_heads, 
                    dropout_p=dropout_p, sr_ratio=sr_ratio)
        self.ffn = mix_feedforward(dim, dim, hidden_features=dim * 4, dropout_p=dropout_p)                    
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path_p) if drop_path_p > 0. else nn.Identity()
    
    def forward(self, x, h, w):
        x1 = x
        x = self.norm1(x)
        x, attn_output = self.attn(x, h, w)
        x = self.drop_path(x)
        x += x1
        x2 = x
        x = self.norm2(x)
        x = self.drop_path(self.ffn(x, h, w))
        x += x2
        return x, attn_output   # attn_output is just for visualization

class mix_transformer_stage(nn.Module):
    def __init__(self, patch_embed, blocks, norm):
        super().__init__()
        self.patch_embed = patch_embed
        self.blocks = blocks
        self.norm = norm

    def forward(self, x):
        x, h, w = self.patch_embed(x)

        for block in self.blocks:
            x, stage_output = block(x, h, w)    # only need the last block for visualization
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        return x, stage_output  # stage_output is just for visualization

class mix_transformer(nn.Module):
    def __init__(self, in_chans, embed_dims, num_heads, depths, 
                sr_ratios, dropout_p, drop_path_p):
        super().__init__()
        self.stages = nn.ModuleList()
        for stage_i in range(len(depths)):
            # Each Stage consists of following blocks :
            # Overlap patch embedding -> mix_transformer_block -> norm
            
            ## TODO: Create a New Block
            if(stage_i == 0):
                patch_size = 7
                stride = 4
                in_chans = in_chans
            else:
                patch_size = 3
                stride = 2
                in_chans = embed_dims[stage_i-1]
            
            patch_embed = overlap_patch_embed(patch_size=patch_size, stride=stride, 
                                              in_chans=in_chans, embed_dim=embed_dims[stage_i])
            blocks = nn.ModuleList()
            for N in range(depths[stage_i]):
                blocks.append(transformer_block(dim=embed_dims[stage_i],
                                                num_heads=num_heads[stage_i],
                                                dropout_p=dropout_p, 
                                                drop_path_p=drop_path_p * (sum(depths[:stage_i])+N) / (sum(depths)-1),    # stochastic depth decay rule
                                                sr_ratio=sr_ratios[stage_i]))
            self.stages.append(mix_transformer_stage(patch_embed, blocks, nn.LayerNorm(embed_dims[stage_i], eps=1e-6)))
            
    def forward(self, x):
        outputs = []
        for stage in self.stages:
            x, _ = stage(x)
            outputs.append(x)
        return outputs

    def get_attn_outputs(self, x):
        # for visualization
        stage_outputs = []
        for stage in self.stages:
            x, stage_data = stage(x)
            stage_outputs.append(stage_data)
        return stage_outputs

class segformer_head(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim, dropout_p=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p

        # 1x1 conv to fuse multi-scale output from encoder
        self.layers = nn.ModuleList([nn.Conv2d(chans, embed_dim, (1, 1)) for chans in in_channels])
        self.linear_fuse = nn.Conv2d(embed_dim * len(self.layers), embed_dim, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(embed_dim, eps=1e-5)
        self.drop = nn.Dropout(dropout_p)

        # 1x1 conv to get num_class channel predictions
        self.linear_pred = nn.Conv2d(self.embed_dim, num_classes, kernel_size=(1, 1))
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.linear_fuse.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        feature_size = x[0].shape[2:]
        # project each encoder stage output to H/4, W/4
        # unify
        unify_stages = [layer(xi) for xi, layer in zip(x, self.layers)]
        # upsampling. The first stage doesn't need upsampling
        ups = [unify_stages[0]] + \
              [F.interpolate(stage, size=feature_size, mode='bilinear') for stage in unify_stages[1:]]

        # concatenate project output and use 1x1
        # convs to get num_class channel output
        concat = torch.cat(ups[::-1], 1)
        x = self.linear_fuse(concat)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.drop(x)
        x = self.linear_pred(x)
        return x

class segformer_mit_b3(nn.Module):    
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Encoder block    
        self.backbone = mix_transformer(in_chans=in_channels, embed_dims=(64, 128, 320, 512), 
                                    num_heads=(1, 2, 5, 8), depths=(3, 4, 18, 3),
                                    sr_ratios=(8, 4, 2, 1), dropout_p=0.1, drop_path_p=0.2)
        # decoder block
        self.decoder_head = segformer_head(in_channels=(64, 128, 320, 512), 
                                    num_classes=num_classes, embed_dim=256)
        
        # init weights
        self.apply(self._init_weights)
         
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                
    def forward(self, x):
        image_hw = x.shape[2:]
        x = self.backbone(x)
        x = self.decoder_head(x)
        # Interpolate to the same as input size
        x = F.interpolate(x, size=image_hw, mode='bilinear')
        return x

    def get_attention_outputs(self, x):
        # contains 4 stages, each stage contains a dictionary, 
        # which contains {'attn':attn}
        return self.backbone.get_attn_outputs(x)    