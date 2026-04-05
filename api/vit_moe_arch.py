"""
ViT + MoE v3 Architecture
Extracted from training/train_vit_moe_v3.py for inference use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class Expert(nn.Module):
    def __init__(self, embed_dim, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * expansion, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MoE_Layer(nn.Module):
    def __init__(self, embed_dim, num_experts=4, k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.router = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList([Expert(embed_dim, dropout=dropout) for _ in range(num_experts)])
        self.aux_loss = 0.0

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        x_flat = x.view(-1, embed_dim)

        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        expert_mask = F.one_hot(top_k_indices, self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)
        router_prob_per_expert = routing_weights.mean(dim=0)
        self.aux_loss = self.num_experts * (tokens_per_expert * router_prob_per_expert).sum()

        out_flat = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            for k_idx in range(self.k):
                mask = (top_k_indices[:, k_idx] == i)
                if not mask.any():
                    continue
                expert_input = x_flat[mask]
                expert_output = expert(expert_input)
                weight = top_k_weights[mask, k_idx].unsqueeze(-1)
                out_flat[mask] += weight * expert_output

        return out_flat.view(batch_size, seq_len, embed_dim)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_experts=4, k=2,
                 attn_dropout=0.1, proj_dropout=0.1, moe_dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.proj_drop = nn.Dropout(proj_dropout)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.moe = MoE_Layer(embed_dim, num_experts=num_experts, k=k, dropout=moe_dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        attn_out = self.proj_drop(attn_out)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.moe(self.norm2(x)))
        return x


class ViT_MoE_v3(nn.Module):
    def __init__(self, num_classes, embed_dim=256, depth=4, num_heads=8,
                 num_experts=4, k=2, patch_size=16, img_size=224,
                 dropout=0.05, drop_path_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        drop_path_rates = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim, num_heads=num_heads, num_experts=num_experts, k=k,
                attn_dropout=dropout, proj_dropout=dropout, moe_dropout=dropout,
                drop_path=drop_path_rates[i],
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_aux_loss(self):
        total = 0.0
        count = 0
        for block in self.blocks:
            total += block.moe.aux_loss
            count += 1
        return total / max(count, 1)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])
        return self.head(x)
